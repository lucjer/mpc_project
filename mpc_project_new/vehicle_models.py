import abc
import numpy as np
import yaml


class MPCModel(metaclass=abc.ABCMeta):
    def __init__(self, model_config_yaml):
        with open(model_config_yaml, 'r') as f:
            data = yaml.safe_load(f)
        self.model_params = data['model_params']
        self.mpc_params = data['mpc_params']
        self.n_states = self.model_params['n_states']
        self.n_inputs = self.model_params['n_inputs']

    @abc.abstractmethod
    def nonlinear_model(self, state, input):
        """Nonlinear vector field. Returns x_dot = f(x, u)."""
        pass

    @abc.abstractmethod
    def linearization_model(self, state, input):
        """Linearized model. Returns linearization (A, B) at operating point (x, u)."""
        pass

    @abc.abstractmethod
    def step_nonlinear_model(self, state, input):
        """Discrete-time nonlinear model. Returns x_(i+1) given x_i and u_i."""
        pass


class UnicycleConstantSpeed(MPCModel):

  def nonlinear_model(self, state, input):
    """
    Calculate the vector field for a nonlinear vehicle model.
    Parameters:
    - state (array-like): The current state of the vehicle, consisting of x, y, and theta coordinates.
    - input (array-like): The input to the vehicle model, consisting of the steering angle delta.
    Returns:
    - state_dot (array-like): The derivative of the state vector, representing the vector field.
    """
    l = self.model_params['l'] 
    v = self.model_params['v']
    x = state[0]
    y = state[1]
    theta = state[2]
    delta = input[0]
    
    state_dot = np.zeros((3, 1))
    x_dot = v * np.cos(theta + delta) 
    y_dot = v * np.sin(theta + delta) 
    theta_dot = v * np.sin(delta)
    state_dot[0] = x_dot
    state_dot[1] = y_dot
    state_dot[2] = theta_dot
    return state_dot


  def linearization_model(self, state, input):
    """
      Linearizes the vehicle model around the given state and input.
    Args:
      state (list): The current state of the vehicle [x, y, theta].
      input (list): The current input to the vehicle [delta].
    Returns:
      tuple: A tuple containing the linearized state matrix A and the linearized input matrix B.
    """
    dt = self.mpc_params['dt']
    x = state[0]
    y = state[1]
    theta = state[2]
    delta = input[0]
    A = np.zeros((3, 3))
    B = np.zeros((3, 1))
    for i in range(3):
      A[i, i] = 1
    A[0, 2] = - np.cos(theta + delta) * dt
    A[1, 2] = np.cos(theta + delta) * dt
    B[2, 0] = np.cos(delta) * dt
    return A, B
  
  def step_nonlinear_model(self, state, input):
    # Euler discretization
    l = self.model_params['l'] 
    v = self.model_params['v'] 
    dt = self.mpc_params['dt']
    
    x = state[0]
    y = state[1]
    theta = state[2]
    delta = input[0]
    new_state = np.zeros((3, 1))
    x_ = x + v * np.cos(theta + delta) * dt
    y_ = y + v * np.sin(theta + delta) * dt
    theta_ = theta + v * np.sin(delta) * dt
    new_state[0] = x_
    new_state[1] = y_
    new_state[2] = theta_
    return new_state
  
  
class KinematicBicycleConstantSpeed(MPCModel):
  def nonlinear_model(self, state, input):
    l_front = self.model_params['l_f']
    l_rear = self.model_params['l_r']
    v = self.model_params['v']
    
    x = state[0]
    y = state[1]
    theta = state[2]
    delta = input[0]
    beta = np.arctan((l_rear / (l_front + l_rear)) * np.tan(delta))
    state_dot = np.zeros((3, ))
    x_dot = v * np.cos(theta + beta)
    y_dot = v * np.sin(theta + beta)
    theta_dot = (v / l_rear) * np.sin(beta)
    
    state_dot[0] = x_dot
    state_dot[1] = y_dot
    state_dot[2] = theta_dot
    
    return state_dot
  
  def linearization_model(self, state, input):
        l_front = self.model_params['l_f']
        l_rear = self.model_params['l_r']
        v = self.model_params['v']
        theta = state[2]
        delta = input[0]
        
        # Calculate beta
        alpha = l_rear / (l_front + l_rear)
        beta = np.arctan(alpha * np.tan(delta))
        d_beta_dtan_delta = 1 / (alpha**2 * np.tan(delta)**2 + 1)
        d_tan_delta_d_delta = 1 / (np.cos(delta)**2) * alpha
        # d_beta_d_delta = (l_rear / (l_front + l_rear)) / (np.cos(delta)**2 + (l_rear / (l_front + l_rear))**2 * np.tan(delta)**2)
        d_beta_d_delta = d_beta_dtan_delta * d_tan_delta_d_delta
        # Jacobian A
        A = np.zeros((3, 3))
        A[0, 2] = -v * np.sin(theta + beta)
        A[1, 2] = v * np.cos(theta + beta)
        
        # Jacobian B
        B = np.zeros((3, 1))
        B[0, 0] = -v * np.sin(theta + beta) * d_beta_d_delta
        B[1, 0] = v * np.cos(theta + beta) * d_beta_d_delta
        B[2, 0] = (v / l_rear) * np.cos(beta) * d_beta_d_delta
        
        # Jacobian C
        C = np.zeros((3, 3))
        return A, B

  def step_nonlinear_model(self, state, input):
    new_state = np.zeros((3, 1))
    l_front = self.model_params['l_f']
    l_rear = self.model_params['l_r']
    v = self.model_params['v']
    x = state[0]
    y = state[1]
    theta = state[2]
    delta = input[0]
    beta = np.arctan((l_rear / (l_front + l_rear)) * np.tan(delta))
    state_dot = np.zeros((3, ))
    x_dot = v * np.cos(theta + beta)
    y_dot = v * np.sin(theta + beta)
    theta_dot = (v / l_rear) * np.sin(beta)
    
    new_state[0] = x_dot * self.mpc_params['dt'] + x
    new_state[1] = y_dot * self.mpc_params['dt'] + y
    new_state[2] = theta_dot * self.mpc_params['dt'] + theta
    return new_state