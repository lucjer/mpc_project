import osqp
import numpy as np
from scipy import sparse
import utils as utils 
import time
import matplotlib.pyplot as plt
import math

# TESTING MORE EFFICIENT
from scipy.sparse import eye, block_diag, csc_matrix, vstack


import cProfile
import pstats
import io



import time
from functools import wraps


def timer_decorator(func):
    call_count = 0
    total_time = 0

    @wraps(func)
    def wrapper(*args, **kwargs):
        nonlocal call_count, total_time
        start_time = time.time()
        result = func(*args, **kwargs)
        end_time = time.time()
        execution_time = end_time - start_time

        call_count += 1
        total_time += execution_time
        average_time = total_time / call_count

        # Convert time to milliseconds and print as integer
        execution_time_ms = int(execution_time * 1000)
        average_time_ms = int(average_time * 1000)

        print(f"Execution time: {execution_time_ms} ms")
        
        if call_count > 1:  # Print average only after the first call
            print(f"Average execution time after {call_count} calls: {average_time_ms} ms")
        
        return result

    return wrapper


class NMPCSolver:
    def __init__(self, model, N, Q, R, QN, alpha, StateConstraints, InputConstraints):
        """
        Generic MPC constructor for SQP optimization
        :param model: dynamical system model
        :param N: horizon | int
        :param nx: state dimension
        :param nu: input dimension
        :param Q: state cost matrix
        :param R: input cost matrix
        :param QN: final state cost matrix        
        :param StateConstraints: dict of state constraints
        :param InputConstraints: dict of input constraints
        """

        # Parameters
        self.N = N # Prediction Horizon
        self.Q = Q  # State cost
        self.R = R  # Input cost
        self.QN = QN # Final state cost
        
        # Model and Dimensions
        self.model = model
        self.nx = model.n_states
        self.nu = model.n_inputs
        
        # Constraints
        self.state_constraints = StateConstraints
        self.input_constraints = InputConstraints       

        # Current state and control
        self.current_prediction = None
        self.current_control = None
        
        # Initialize the optimizer
        self.optimizer = osqp.OSQP()
        print("MPC Class initialized successfully")


    def profile_solve_sqp(self, *args, **kwargs):
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
        U_ref = U_ref
        u_min = self.input_constraints['umin'] 
        u_max = self.input_constraints['umax']
        u_min_array = np.tile(u_min, self.N) + np.array(U_ref).reshape(self.N * self.nu, )
        u_max_array = np.tile(u_max, self.N) + np.array(U_ref).reshape(self.N * self.nu, )
        return u_min_array, u_max_array
    
    @timer_decorator
    def solve_sqp(self, current_state, X_ref, U_ref, X_guess=None, U_guess=None, debug=False, sqp_iter=1, alpha=1.0):
        """
        Initialize the optimization problem.
        :param X: current reference state
        :param U: current control
        """
        # Initialize guess for the optimization problem - SQP
        if X_guess is None or U_guess is None: # If no guess is provided
            if self.current_control is not None and self.current_prediction is not None:
                U_guess = np.zeros_like(self.current_control)
                X_guess = np.zeros_like(self.current_prediction)
                U_guess[0:-1] = self.current_control[1:]
                U_guess[-1] = self.current_control[-1] 
                X_guess[0:-1] = self.current_prediction[1:]
                X_guess[-1] = self.current_prediction[-1]
            else:
                X_guess = X_ref.copy()
                U_guess = U_ref.copy()
        

        X_guess_evolution = [X_guess.copy()]
        U_guess_evolution = [U_guess.copy()]
        cost_evolution = []

        # Initialize matrices that do not change
        I_Nx = sparse.eye(self.N * self.nx, format='csc')
        P_Q = sparse.kron(sparse.eye(self.N, format='csc'), self.Q, format='csc')
        P_R = sparse.kron(sparse.eye(self.N, format='csc'), self.R, format='csc')
        P = sparse.block_diag([P_Q, P_R], format='csc')

        # Initialize matrices that change
        A_states = np.zeros((self.nx * self.N, self.nx * self.N))
        B_states = np.zeros((self.nx * self.N, self.nu * self.N))

        # Use a more efficient sparse format for matrix construction
        Aeq = sparse.lil_matrix((self.nx * self.N, self.nx * self.N + self.nu * self.N))

        # Iterate over horizon
        for j in range(sqp_iter):

            r = np.zeros(self.nx * self.N)

            # First order correction
            step_nonlinear_result = np.array(self.model.step_nonlinear_model(current_state, U_guess[0]))
            r[0:self.nx] = step_nonlinear_result - X_guess[0]

            for k in range(self.N - 1):
                state_guess_k = X_guess[k]
                input_guess_k = U_guess[k]

                A_lin, B_lin, _, _ = self.model.linearization_model(state_guess_k, input_guess_k)
                A_states[(k + 1) * self.nx: (k + 2) * self.nx, k * self.nx: (k + 1) * self.nx] = A_lin
                B_states[(k + 1) * self.nx: (k + 2) * self.nx, k * self.nu: (k + 1) * self.nu] = B_lin 

                r[(k + 1) * self.nx: (k + 2) * self.nx] = self.model.step_nonlinear_model(state_guess_k, input_guess_k) - X_guess[k+1]

            # Compute Aeq and Aineq
            Ax = -I_Nx + sparse.csc_matrix(A_states)
            Bu = sparse.csc_matrix(B_states)
            
            # Directly assign to preallocated Aeq
            Aeq[:, :self.nx * self.N] = Ax
            Aeq[:, self.nx * self.N:] = Bu
            Aeq = Aeq.tocsc()

            # Construct Aineq and the full constraint matrix
            Aineq = sparse.eye(self.N * self.nx + self.N * self.nu, format='csc')
            A_states_full = sparse.vstack([Aeq, Aineq], format='csc')

            # Constraints
            u_min, u_max = self.compute_input_constraints(-U_guess)
            lineq = np.hstack([-np.inf] * self.N * self.nx + list(u_min))
            uineq = np.hstack([np.inf] * self.N * self.nx + list(u_max))
            leq = -r
            ueq = -r
            l = np.hstack([leq, lineq])
            u = np.hstack([ueq, uineq])

            # Objective function
            xug = np.hstack([np.concatenate(X_guess), np.concatenate(U_guess)])
            xur = np.hstack([np.concatenate(X_ref), np.concatenate(U_ref)])
            delta_X = xug - xur
            q = P.dot(delta_X)

            # Solve quadratic program
            self.optimizer = osqp.OSQP()
            self.optimizer.setup(P=P, q=q, A=A_states_full, l=l, u=u, verbose=False)
            dec = self.optimizer.solve()
            delta_U = np.reshape(dec.x[-self.N * self.nu:], (self.N, self.nu))
            delta_X = np.reshape(dec.x[:self.N * self.nx], (self.N, self.nx))

            # Update guesses
            X_guess += alpha * delta_X
            U_guess += alpha * delta_U

            if debug:
                X_guess_evolution.append(X_guess.copy())
                U_guess_evolution.append(U_guess.copy())

        if debug:
            utils.plot_xref_evolution(X_guess_evolution, filename="evolution_sqp.pdf")
            print("SAVING EVOLUTION")
        self.current_control = U_guess
        self.current_prediction = X_guess
        return X_guess, U_guess



        
if False:       
    class NMPCSolverOutput():
        def __init__(self, model, N, Q, R, QN, alpha, StateConstraints, InputConstraints):
            """
            Generic MPC constructor for SQP optimization
            :param model: dynamical system model
            :param N: horizon | int
            :param nx: state dimension
            :param nu: input dimension
            :param Q: state cost matrix
            :param R: input cost matrix
            :param QN: final state cost matrix        
            :param StateConstraints: dict of state constraints
            :param InputConstraints: dict of input constraints
            """

            # Parameters
            self.N = N # Prediction Horizon
            self.Q = Q  # State cost
            self.R = R  # Input cost
            self.QN = QN # Final state cost        
            
            # Model and Dimensions
            self.model = model
            self.nx = model.n_states
            self.nu = model.n_inputs
            self.ny = model.n_outputs
            
            # Constraints
            self.state_constraints = StateConstraints
            self.input_constraints = InputConstraints
            self.output_constraints = None       
            
            # Initialize the optimizer
            self.optimizer = osqp.OSQP()
            print("MPC Class initialized successfully")
            
            
        @timer_decorator
        def solve_sqp(self, current_state, X_ref, U_ref, debug=False, sqp_iter=1, alpha=1.0):
            """
            Initialize the optimization problem.
            :param X: current reference state
            :param U: current control
            """
            # Initialize guess for the optimization problem - SQP
            X_guess = X_ref.copy()
            U_guess = U_ref.copy()
            
            # To store the evolution of X_ref over iterations
            X_guess_evolution = [X_guess.copy()]
            U_guess_evolution = [U_guess.copy()]
            cost_evolution = []

            # Iterate over horizon
            for _ in range(sqp_iter):
                # LTV System Matrices
                A_states = np.zeros((self.nx * (self.N), self.nx * (self.N)))
                B_states = np.zeros((self.nx * (self.N), self.nu * (self.N)))
                C_states = np.zeros((self.ny * (self.N), self.nx * (self.N)))
                D_states = np.zeros((self.ny * (self.N), self.nu * (self.N)))
                
                # First order correction
                r_x = np.zeros((self.nx * (self.N), ))
                r_y = np.zeros((self.ny * (self.N), ))
                
                r_x[0: self.nx] = self.model.step_nonlinear_model(current_state, U_guess[0]) - X_guess[0]
                r_y[0: self.ny] = self.model.output_model(current_state, X_guess[0], U_guess[0])
                
                for k in range(self.N - 1):
                    # Fetch current guess at step k for state and input
                    state_guess_k = X_guess[k]
                    input_guess_k = U_guess[k]
                    reference_state_k = X_ref[k]
                    reference_input_k = U_ref[k]
                    
                    
                    # Compute LTV matrices - Linearization at current time step
                    A_lin, B_lin, C_lin, D_lin = self.model.linearization_model(state_guess_k,
                                                                                input_guess_k,
                                                                                reference_state_k,
                                                                                reference_input_k)
                    A_states[(k + 1) * self.nx: (k + 2) * self.nx, k * self.nx:(k + 1) * self.nx] = A_lin
                    B_states[(k + 1) * self.nx: (k + 2) * self.nx, k * self.nu:(k + 1) * self.nu] = B_lin
                    C_states[k * self.ny:(k + 1) * self.ny, k * self.nx:(k + 1) * self.nx] = C_lin
                    D_states[k * self.ny:(k + 1) * self.ny, k * self.nu:(k + 1) * self.nu] = D_lin
                    
                    # Compute the first order correction
                    r_x[(k + 1) * self.nx: (k + 2) * self.nx] = self.model.step_nonlinear_model(state_guess_k, input_guess_k) - X_guess[k+1] 
                    r_y[(k + 1) * self.ny: (k + 2) * self.ny] = self.model.output_model(current_state, state_guess_k, input_guess_k)
                    
                Ax = sparse.kron(sparse.eye(self.N), - sparse.eye(self.nx)) + sparse.csc_matrix(A_states)
                Bu = sparse.csc_matrix(B_states)
                Aeqx = sparse.hstack([Ax, sparse.csr_matrix((self.nx * self.N, self.ny * self.N)), Bu])
                Aeqy = sparse.hstacl([C_states, sparse.csr_matrix((self.ny * self.N, self.nx * self.N)), D_states])
                Aeq = sparse.vstack([Aeqx, Aeqy], format='csc')
                
                Aineq = sparse.eye((self.N) * self.nx + self.N * self.nu)
                A_states = sparse.vstack([Aeq, Aineq], format='csc')

                lineq = np.hstack([-np.inf, -np.inf, -np.inf] * (self.N) + [-math.pi/12] * self.N)
                uineq = np.hstack([np.inf, np.inf, np.inf] * (self.N) + [math.pi/12] * self.N)
                leq = - r_x
                ueq = - r_x
                l = np.hstack([leq, lineq])
                u = np.hstack([ueq, uineq])

                P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q),
                    sparse.kron(sparse.eye(self.N), self.R)], format='csc')
                
                xug = np.hstack([np.concatenate(X_guess), np.concatenate(U_guess)])
                xur = np.hstack([np.concatenate(X_ref), np.concatenate(U_ref)])
                delta_X = xug - xur      
                q = P.dot(delta_X)

                self.optimizer = osqp.OSQP()
                self.optimizer.setup(P=P, q=q, A=A_states, l=l, u=u, verbose=False)
                dec = self.optimizer.solve()
                delta_U = np.array(dec.x[-self.N * self.nu:])
                delta_X = np.reshape(dec.x[:self.N * self.nx], (self.N, self.nx))
                
                # Update guesses
                for i in range(self.N):
                    print(delta_X[i])
                    X_guess[i] = X_guess[i] + delta_X[i] * alpha
                for i in range(self.N):
                    U_guess[i] = U_guess[i] + delta_U[i] * alpha
                
                # Store the current X_guess to track evolution
                if debug:
                    X_guess_evolution.append(X_guess.copy())
                    U_guess_evolution.append(U_guess.copy())
                    #cost_evolution.append(self.compute_cost(delta_X, delta_U))

            if debug:
                utils.plot_xref_evolution(X_guess_evolution, filename="evolution_sqp.pdf")
                # TODO: Implement cost evolution - to check convergence
                # utils.plot_cost_evolution(cost_evolution, filename="cost_evolution_sqp.pdf")
                
            return X_guess, U_guess
        
        def compute_cost_evolution(self, U_guess, current_state, X_ref):
            pass