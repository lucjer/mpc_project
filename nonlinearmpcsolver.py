import osqp
import numpy as np
from scipy import sparse
import utils as utils 
import time
import matplotlib.pyplot as plt
import math

# Profiling
import cProfile
import pstats
import io



import time
from functools import wraps


class NMPCSolver:
    def __init__(self, model, N, Q, R, StateConstraints, InputConstraints, sqp_iters=1, alpha=0.1, debug_plotting_callback=None, debug_plots_folder=None):
        """
        Generic MPC constructor for SQP optimization
        :param model: dynamical system model
        :param N: horizon length
        :param nx: state dimension
        :param nu: input dimension
        :param Q: state cost matrix
        :param R: input cost matrix
        :param StateConstraints: dict of state constraints
        :param InputConstraints: dict of input constraints
        :oaram debug_plotting_callback: callback function for debugging
        :param debug_plots_folder: folder to save debug plots
        """

        # Parameters
        self.N = N # Prediction Horizon
        self.Q = Q  # State cost
        self.R = R  # Input cost
        
        # Model and Dimensions
        self.model = model
        self.nx = model.n_states
        self.nu = model.n_inputs
        
        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints       

        # Current state and control
        self.current_state_guess = None
        self.current_input_guess = None
        
        # Initialize the optimizer
        self.optimizer = osqp.OSQP()
        self.sqp_iters = sqp_iters
        self.alpha = alpha

        # Debugging parameters - Callback function and folder to save plots
        self.debug_plotting_callback = debug_plotting_callback
        self.debug_plots_folder = debug_plots_folder


    def profile_solve_sqp(self, *args, **kwargs): 
        # Used for profiling. Useful for investigating performance bottlenecks
        profiler = cProfile.Profile()
        profiler.enable()

        result = self.solve_sqp(*args, **kwargs)

        profiler.disable()
        profiler.dump_stats('solve_sqp_profile.prof')

        return result
    
    def compute_input_constraints(self, U_ref):
        """
        Compute the input constraints for the optimization problem
        :param U_guess: current control guess
        :return: input constraints
        """
        u_min = self.input_constraints['umin'] 
        u_max = self.input_constraints['umax']
        u_min_array = np.tile(u_min, self.N) - np.array(U_ref).reshape(self.N * self.nu, )
        u_max_array = np.tile(u_max, self.N) - np.array(U_ref).reshape(self.N * self.nu, )
        return u_min_array, u_max_array
    
    def solve_sqp(self, current_state, X_ref, U_ref, X_guess=None, U_guess=None, debug=False):
        """
        Initialize the optimization problem.
        :param X: current reference state
        :param U: current control
        """
        # Initialize guess for the optimization problem - SQP
        if X_guess is None or U_guess is None: # If no guess is provided
            if self.current_input_guess is not None and self.current_state_guess is not None:
                # TODO: Improve this initialization
                U_guess = np.zeros_like(self.current_input_guess)
                X_guess = np.zeros_like(self.current_state_guess)
                # Initialize guess with the previous solution
                U_guess[0:-1] = self.current_input_guess[1:]
                X_guess[0:-1] = self.current_state_guess[1:]
                U_guess[-1] = self.current_input_guess[-1] 
                # Compute the final state by re-applying the last input (naif solution)
                X_guess[-1] = self.model.step_nonlinear_model(self.current_state_guess[-1], self.current_input_guess[-1])
            else:
                X_guess = X_ref.copy()
                U_guess = U_ref.copy()
        
        # Save evolution of the guesses for debugging
        if debug:
            X_guess_evolution = [X_guess.copy()]
            U_guess_evolution = [U_guess.copy()]
        
        # Initialize matrices that do not change
        I_Nx = sparse.eye(self.N * self.nx, format='csc')
        P_Q = sparse.kron(sparse.eye(self.N, format='csc'), self.Q, format='csc')
        P_R = sparse.kron(sparse.eye(self.N, format='csc'), self.R, format='csc')
        P = sparse.block_diag([P_Q, P_R], format='csc')

        # Initialize only once matrices for dynamic constraints 
        A_states = np.zeros((self.nx * self.N, self.nx * self.N))
        B_states = np.zeros((self.nx * self.N, self.nu * self.N))
        Aeq = sparse.lil_matrix((self.nx * self.N, self.nx * self.N + self.nu * self.N))
        r = np.zeros(self.nx * self.N)

        for j in range(self.sqp_iters):
            # 1) Dynamics Constraints 
            # Initialize residuals 
            step_nonlinear_result = np.array(self.model.step_nonlinear_model(current_state, U_guess[0]))
            r[0:self.nx] = step_nonlinear_result - X_guess[0]

            for k in range(self.N - 1):
                state_guess_k = X_guess[k]
                input_guess_k = U_guess[k]

                # Compute linearization at step k
                A_lin, B_lin, _, _ = self.model.linearization_model(state_guess_k, input_guess_k)
                A_states[(k + 1) * self.nx: (k + 2) * self.nx, k * self.nx: (k + 1) * self.nx] = A_lin
                B_states[(k + 1) * self.nx: (k + 2) * self.nx, k * self.nu: (k + 1) * self.nu] = B_lin 

                r[(k + 1) * self.nx: (k + 2) * self.nx] = self.model.step_nonlinear_model(state_guess_k, input_guess_k) - X_guess[k+1]

            # Compute Aeq: AX + BU + r = X
            Ax = - I_Nx + sparse.csc_matrix(A_states)
            Bu = sparse.csc_matrix(B_states)
            # Update Aeq for dynamics equality constraints
            Aeq[:, :self.nx * self.N] = Ax
            Aeq[:, self.nx * self.N:] = Bu
            Aeq = Aeq.tocsc()
            leq = - r
            ueq = - r

            # 2) Inequality Constraints on states and inputs
            Aineq = sparse.eye(self.N * self.nx + self.N * self.nu, format='csc')
            # Compute input constraints
            # TODO: Input inequality constraints only implemented for the linear case
            u_min, u_max = self.compute_input_constraints(U_guess)
            # TODO: State inequality constraints not implemented yet
            states_min = [-np.inf] * self.N * self.nx 
            states_max = [np.inf] * self.N * self.nx 
            # Combine input and state constraints
            lineq = np.hstack(states_min + list(u_min))
            uineq = np.hstack(states_max + list(u_max))

            # TODO: 3) Add implementation for constraints on variations of input signals 
            

            l = np.hstack([leq, lineq])
            u = np.hstack([ueq, uineq])

            A_full = sparse.vstack([Aeq, Aineq], format='csc')


            # Objective function
            xug = np.hstack([np.concatenate(X_guess), np.concatenate(U_guess)])
            xur = np.hstack([np.concatenate(X_ref), np.concatenate(U_ref)])
            delta_X = xug - xur
            q = P.dot(delta_X)

            # Solve quadratic program
            self.optimizer = osqp.OSQP()
            self.optimizer.setup(P=P, q=q, A=A_full, l=l, u=u, verbose=False)
            dec = self.optimizer.solve()
            delta_U = np.reshape(dec.x[-self.N * self.nu:], (self.N, self.nu))
            delta_X = np.reshape(dec.x[:self.N * self.nx], (self.N, self.nx))

            # Update guesses
            X_guess += self.alpha * delta_X
            U_guess += self.alpha * delta_U

            if debug:
                X_guess_evolution.append(X_guess.copy())
                U_guess_evolution.append(U_guess.copy())

        if debug:
            results = {'X_guess': X_guess, 'U_guess': U_guess, 'X_guess_evolution': X_guess_evolution, 'U_guess_evolution': U_guess_evolution}
            self.debug_plotting_callback(current_state, results, folder_path=self.debug_plots_folder)

        # Update current state and control guesses for next call to the solver    
        self.current_input_guess = U_guess
        self.current_state_guess = X_guess
        return X_guess, U_guess

