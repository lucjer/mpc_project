import numpy as np

class MPCController:
    def __init__(self, model, Q, R, S, n_horizon):
      self.model = model
      self.Q = Q
      self.R = R
      self.S = S
      self.n_horizon = n_horizon
      self.previous_state = None # Collects previous ot
      self.previous_input = None


    def get_gamma_lambda_x(self, x_operating, u_operating, n_horizon):
        """
        x_operating: vector of current operating states (n_horizon - dimensional vector of n_states vectors)
        u_operating: vector of current operating inputs (n_horizon - dimensional vector of n_inputs vectors)
        n_horizon: length of the prediction horizon for MPC formulation
        model: nonlinear function (function of state and input - f(x, u))
        """
        n_states = self.model.n_states
        n_inputs = self.model.n_inputs
        n_horizon = self.n_horizon
        gamma_x = np.zeros((n_states * n_horizon, 1))
        lambda_x = np.zeros((n_states * n_horizon, n_inputs * n_horizon))
        A_0, B_0 = self.get_linearization_at(x_operating[0:n_states, :],
                                             u_operating[0:n_inputs, :])
        l_0 = self.get_l_inhomogeneous_term(x_operating[0:n_states, :],
                                            u_operating[0:n_inputs, :],
                                            A_0, B_0, None, 0)
        gamma_i = l_0
        gamma_x[0:n_states] = gamma_i
        lambda_x[0:n_states, 0:n_inputs] = B_0

        for i in range(1, n_horizon):
            A_i, B_i = self.get_linearization_at(x_operating[i*n_states:(i+1)* n_states, :],
                                                 u_operating[i*n_inputs:(i+1)* n_inputs, :])
            l_i = self.get_l_inhomogeneous_term(x_operating[i*n_states:(i+1)* n_states, :],
                                                u_operating[i*n_inputs:(i+1)* n_inputs, :],
                                                A_i,
                                                B_i, None, i)
            gamma_i = np.matmul(A_i, gamma_i) + l_i
            gamma_x[i*n_states:(i+1)*n_states] = gamma_i
            lambda_x[i*n_states:(i+1)*n_states, i*n_inputs:(i+1)*n_inputs] = B_i
            for j in range(i):
                lambda_x[i*n_states:(i+1)*n_states, j*n_inputs:(j+1)*n_inputs] = \
                np.matmul(A_i, lambda_x[(i-1)*n_states:(i)*n_states, j*n_inputs:(j+1)*n_inputs])
        return gamma_x, lambda_x


    def get_l_inhomogeneous_term(self, x_star, u_star, A_i, B_i, model, i):
          if i == 0:
              return self.model.step_nonlinear_model(x_star, u_star) - \
              np.matmul(B_i, u_star)
          else:
              return self.model.step_nonlinear_model(x_star, u_star) - \
              np.matmul(B_i, u_star) - np.matmul(A_i, x_star)


    def get_linearization_at(self, x, u):
        A, B = self.model.linearization_model(x, u)
        return A, B


    def get_gamma_lambda_y(self, x_operating, u_operating, n_horizon):
        return self.get_gamma_lambda_x(x_operating, u_operating, n_horizon)


    def build_hessian_f(self, Q_state, R_input, S_input_variation,
                      lambda_y, gamma_y, y_ref,
                      n_horizon):
        Q_np = np.kron(np.eye(n_horizon), self.Q)
        R_np = np.kron(np.eye(n_horizon), self.R)
        S_np = np.kron(np.eye(n_horizon), self.S)
        # H = 2 * (lambda_y.transpose() @ Q_np @ lambda_y)
        H = 2 * (np.matmul(np.matmul(lambda_y.transpose(), Q_np), lambda_y) + R_np)
        f = 2 * np.matmul(np.matmul((gamma_y - y_ref).transpose(), Q_np), lambda_y)
        if False:
          print("Q_np ")
          print(Q_np)
          print("R_np ")
          print(R_np)
          print("S_np ")
          print(S_np)
          print("lambda_y")
          print(lambda_y)
          print("gamma_y")
          print(gamma_y)
          print("y_ref")
          print(y_ref)
          print("H ")
          print(H)
          print("f")
          print(f)

        return H, f