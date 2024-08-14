# This class is loosely inspired by matsseig MPC implementation
# https://github.com/matssteinweg/Multi-Purpose-MPC/blob/master/src/MPC.py
# Important reference used for the implementation of the MPC controller
# https://cse.lab.imtlucca.it/~bemporad/publications/papers/ijc_rtiltv.pdf

import osqp
import numpy as np
from scipy import sparse
import mpc_project_new.utils as utils 
import matplotlib.pyplot as plt



class GenericNMPC:
    def __init__(self, model, N, Q, R, QN, StateConstraints, InputConstraints):
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
        self.current_control = np.zeros((self.nu * self.N))
        
        # Initialize the optimizer
        self.optimizer = osqp.OSQP()
        print("MPC Class initialized successfully")
    
    def solve_sqp(self, current_state, X_ref, U_ref, debug=False, sqp_iter=1):
        """
        Initialize the optimization problem.
        :param X: current reference state
        :param U: current control
        """
        # Initialize guess for the optimization problem - SQP
        X_ref.append(current_state)
        X_guess = X_ref.copy()
        U_guess = U_ref.copy()
        
        # To store the evolution of X_ref over iterations
        X_ref_evolution = [X_guess.copy()]

        # Iterate over horizon
        for j in range(sqp_iter):
            # LTV System Matrices
            A = np.zeros((self.nx * (self.N + 1), self.nx * (self.N + 1)))
            B = np.zeros((self.nx * (self.N + 1), self.nu * (self.N)))
            
            # First order correction
            r = np.zeros((self.nx * (self.N + 1), ))
            
            r[0: self.nx] = (current_state - X_guess[0])  
            for k in range(self.N):
                # Fetch current guess at step k for state and input
                state_guess_k = X_guess[k]
                input_guess_k = U_guess[k]
                
                # Compute LTV matrices - Linearization at current time step
                A_lin, B_lin = self.model.linearization_model(state_guess_k, input_guess_k)
                A[(k + 1) * self.nx: (k + 2) * self.nx, k * self.nx:(k + 1) * self.nx] = A_lin
                B[(k + 1) * self.nx: (k + 2) * self.nx, k * self.nu:(k + 1) * self.nu] = B_lin 
                
                # Compute the first order correction
                r[(k + 1) * self.nx: (k + 2) * self.nx] = self.model.step_nonlinear_model(state_guess_k, input_guess_k) - X_guess[k+1] 
            
            Ax = sparse.kron(sparse.eye(self.N + 1), - sparse.eye(self.nx)) + sparse.csc_matrix(A)
            Bu = sparse.csc_matrix(B)
            Aeq = sparse.hstack([Ax, Bu])
            Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)
            A = sparse.vstack([Aeq, Aineq], format='csc')

            lineq = np.hstack([-np.inf, -np.inf, -np.inf] * (self.N + 1) + [-np.inf] * self.N)
            uineq = np.hstack([np.inf, np.inf, np.inf] * (self.N + 1) + [np.inf] * self.N)
            leq = - r
            ueq = - r
            l = np.hstack([leq, lineq])
            u = np.hstack([ueq, uineq])

            P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.QN,
                sparse.kron(sparse.eye(self.N), self.R)], format='csc')
            
            xug = np.hstack([np.concatenate(X_guess), np.concatenate(U_guess)])
            xur = np.hstack([np.concatenate(X_ref), np.concatenate(U_ref)])
            delta_X = xug - xur      
            q = P.dot(delta_X)

            self.optimizer = osqp.OSQP()
            self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)
            dec = self.optimizer.solve()
            delta_u = np.array(dec.x[-self.N * self.nu:])
            delta_x = np.reshape(dec.x[:(self.N+1) * self.nx], (self.N+1, self.nx))
            
            print("U_guess")
            print(U_guess)
            input()
            
            # Update guesses
            for i in range(self.N + 1):
                X_guess[i] = X_guess[i] + delta_x[i] * 0.5
            for i in range(self.N):
                U_guess[i] = U_guess[i] + delta_u[i] * 0.5
            
            # Store the current X_guess to track evolution
            X_ref_evolution.append(X_guess.copy())

        # Plotting X_ref evolution
        utils.plot_xref_evolution(X_ref_evolution, filename="evolution_sqp.pdf")

        return X_guess, U_guess

           


        
        

