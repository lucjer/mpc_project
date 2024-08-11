# This class is loosely inspired by matsseig MPC implementation
# https://github.com/matssteinweg/Multi-Purpose-MPC/blob/master/src/MPC.py
# Important reference used for the implementation of the MPC controller
# https://cse.lab.imtlucca.it/~bemporad/teaching/mpc/imt/2-ltv_nl_mpc.pdf
# 

import osqp
import numpy as np
from scipy import sparse


class GenericNMPC:
    def __init__(self, model, Q, R, QN, StateConstraints, InputConstraints):
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
        self.N = model.parameters['N'] # Prediction Horizon
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
    
    def _init_problem(self, current_state, X_ref, U_ref):
        """
        Initialize the optimization problem.
        :param X: current reference state
        :param U: current control
        """
        # Initialize guess for the optimization problem - SQP
        X_guess = X_ref.copy()
        U_guess = U_ref.copy()
        
        # Constraints
        umin = self.input_constraints['umin']
        umax = self.input_constraints['umax']
        dxmin = self.state_constraints['xmin']
        dxmax = self.state_constraints['xmax']

        # LTV System Matrices
        A = np.zeros((self.nx * (self.N + 1), self.nx * (self.N + 1)))
        B = np.zeros((self.nx * (self.N + 1), self.nu * (self.N)))
        
        # First order correction
        r = np.zeros((self.nx * (self.N )))
        r[0: self.nx] = current_state - X_guess[0]
        
        # Iterate over horizon
        sqp_steps = 10
        for j in range(sqp_steps):
            for k in range(self.N):
                # Fetch current guess at step k for state and input
                state_guess_k = X_guess[k]
                input_guess_k = U_guess[k]
                
                # Compute LTV matrices - Linearization at current time step
                A_lin, B_lin = self.model.linearize(state_guess_k, input_guess_k)
                A[(k + 1) * self.nx: (k + 2) * self.nx, k * self.nx:(k + 1) * self.nx] = A_lin
                B[(k + 1) * self.nx: (k + 2) * self.nx, k * self.nu:(k + 1) * self.nu] = B_lin
                
                # Compute the first order correction
                r[(k + 1) * self.nx: (k + 2) * self.nx] = X_guess[k] - self.model.dynamics(state_guess_k, input_guess_k)
            


        # Get equality matrix - Dynamics Constraints 
        Ax = sparse.kron(sparse.eye(self.N + 1), - sparse.eye(self.nx)) + sparse.csc_matrix(A)
        Bu = sparse.csc_matrix(B)
        Aeq = sparse.hstack([Ax, Bu])
        
        # Get inequality matrix
        Aineq = sparse.eye((self.N + 1) * self.nx + self.N * self.nu)
        # Combine constraint matrices
        A = sparse.vstack([Aeq, Aineq], format='csc')

        # Get upper and lower bound vectors for equality constraints
        lineq = np.hstack([xmin_dyn,
                           np.kron(np.ones(self.N), umin)])
        uineq = np.hstack([xmax_dyn, umax_dyn])
        # Get upper and lower bound vectors for inequality constraints
        leq = np.hstack([-x0, uq])
        ueq = leq
        # Combine upper and lower bound vectors
        l = np.hstack([leq, lineq])
        u = np.hstack([ueq, uineq])

        # Set cost matrices
        P = sparse.block_diag([sparse.kron(sparse.eye(self.N), self.Q), self.QN,
             sparse.kron(sparse.eye(self.N), self.R)], format='csc')
        q = np.hstack(
            [-np.tile(np.diag(self.Q.A), self.N) * xr[:-self.nx],
             -self.QN.dot(xr[-self.nx:]),
             -np.tile(np.diag(self.R.A), self.N) * ur])

        # Initialize optimizer
        self.optimizer = osqp.OSQP()
        self.optimizer.setup(P=P, q=q, A=A, l=l, u=u, verbose=False)


        
        

