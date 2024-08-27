import abc
import numpy as np
import yaml
import math


class MPCModel(metaclass=abc.ABCMeta):
    """ Abstract class for MPC models. """
    def __init__(self, model_config_yaml):
        with open(model_config_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.model_params = data["model_params"]
        self.mpc_params = data["mpc_params"]
        self.n_states = self.model_params["n_states"]
        self.n_inputs = self.model_params["n_inputs"]

    @abc.abstractmethod
    def nonlinear_model(self, state, inp):
        """Nonlinear vector field. Returns x_dot = f(x, u)."""

    @abc.abstractmethod
    def linearization_model(self, state, inp, reference_state, reference_input):
        """Linearized model around the given state, input, reference"""

    @abc.abstractmethod
    def step_nonlinear_model(self, state, inp):
        """Discrete-time nonlinear model. Returns x_(i+1) given x_i and u_i."""

    @abc.abstractmethod
    def output_model(self, state, inp):
        """
        Output model for the vehicle.
        """


class UnicycleConstantSpeed(MPCModel):
    """ Unicycle model with constant speed. """
    def nonlinear_model(self, state, inp):
        """
        Calculate the vector field for the unicle vehicle model.
        Parameters:
        - state (array-like): The current state of the vehicle, consisting of x, y, and theta coordinates.
        - inp (array-like): The input to the vehicle model, consisting of the steering angle delta.
        Returns:
        - state_dot (array-like): The derivative of the state vector, representing the vector field.
        """
        v = self.model_params["v"]
        theta = state[2]
        delta = inp[0]

        state_dot = np.zeros((3, 1))
        x_dot = v * math.cos(theta + delta)
        y_dot = v * math.sin(theta + delta)
        theta_dot = v * math.sin(delta)
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
        dt = self.mpc_params["dt"]
    
        theta = state[2]
        delta = input[0]
        A = np.zeros((3, 3))
        B = np.zeros((3, 1))
        for i in range(3):
            A[i, i] = 1
        A[0, 2] = -math.cos(theta + delta) * dt
        A[1, 2] = math.cos(theta + delta) * dt
        B[2, 0] = math.cos(delta) * dt
        return A, B

    def step_nonlinear_model(self, state, input):
        # Euler discretization
        v = self.model_params["v"]
        dt = self.mpc_params["dt"]

        x = state[0]
        y = state[1]
        theta = state[2]
        delta = input[0]
        new_state = np.zeros((3, 1))
        x_ = x + v * math.cos(theta + delta) * dt
        y_ = y + v * math.sin(theta + delta) * dt
        theta_ = theta + v * math.sin(delta) * dt
        new_state[0] = x_
        new_state[1] = y_
        new_state[2] = theta_
        return new_state


class KinematicBicycleConstantSpeed(MPCModel):
    def nonlinear_model(self, state, input):
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        v = self.model_params["v"]

    
        theta = state[2]
        delta = input[0]
        beta = np.arctan((l_rear / (l_front + l_rear)) * np.tan(delta))
        state_dot = np.zeros((3,))
        x_dot = v * math.cos(theta + beta)
        y_dot = v * math.sin(theta + beta)
        theta_dot = (v / l_rear) * math.sin(beta)

        state_dot[0] = x_dot
        state_dot[1] = y_dot
        state_dot[2] = theta_dot

        return state_dot

    def linearization_model(
        self, state, input, reference_state=None, reference_input=None, debug=False
    ):
        """
        Linearizes the vehicle model around the given state and input.

        Args:
            state (list): The current state of the vehicle [x, y, theta].
            input (list): The current input to the vehicle [delta].
            debug (bool): If True, prints the matrices A, B, and C.

        Returns:
            tuple: A tuple containing the linearized state matrix A, the linearized input matrix B, and matrix C.
        """
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        v = self.model_params["v"]
        theta = state[2]
        delta = input

        # Calculate beta
        alpha = l_rear / (l_front + l_rear)
        beta = np.arctan(alpha * np.tan(delta))
        d_beta_dtan_delta = 1 / (alpha**2 * np.tan(delta) ** 2 + 1)
        d_tan_delta_d_delta = 1 / (math.cos(delta) ** 2) * alpha
        d_beta_d_delta = d_beta_dtan_delta * d_tan_delta_d_delta

        # Jacobian A
        A = np.zeros((3, 3))
        for i in range(3):
            A[i, i] = 1
        A[0, 2] = -v * math.sin(theta + beta) * self.mpc_params["dt"]
        A[1, 2] = v * math.cos(theta + beta) * self.mpc_params["dt"]

        # Jacobian B
        B = np.zeros((3, 1))
        B[0, 0] = -v * math.sin(theta + beta) * d_beta_d_delta * self.mpc_params["dt"]
        B[1, 0] = v * math.cos(theta + beta) * d_beta_d_delta * self.mpc_params["dt"]
        B[2, 0] = (v / l_rear) * math.cos(beta) * d_beta_d_delta * self.mpc_params["dt"]

        # Jacobian C
        C = np.zeros((3, 3))

        if debug:
            print("Linearized State Matrix A:")
            print(A)
            print("\nLinearized Input Matrix B:")
            print(B)
            print("\nMatrix C:")
            print(C)

        return A, B, C, B

    def output_model(self, state, input):
        return state

    def step_nonlinear_model(self, state, input, debug=False):
        if debug:
            return state
        new_state = np.zeros((3,))
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        v = self.model_params["v"]
        x = state[0]
        y = state[1]
        theta = state[2]
        delta = input
        beta = np.arctan((l_rear / (l_front + l_rear)) * np.tan(delta))
        x_dot = v * math.cos(theta + beta)
        y_dot = v * math.sin(theta + beta)
        theta_dot = (v / l_rear) * math.sin(beta)

        new_state[0] = x_dot * self.mpc_params["dt"] + x
        new_state[1] = y_dot * self.mpc_params["dt"] + y
        new_state[2] = theta_dot * self.mpc_params["dt"] + theta
        return new_state


class KinematicBicycleSpatialSpeed(MPCModel):
    def nonlinear_model(self, state, input):
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]

        # STATES
        theta = state[2]
        v = state[3]

        # INPUTS
        delta = input[0]
        a = input[1]
        s_dot = input[2]

        # Update the state vector
        state_dot = np.zeros((5, 1))
        beta = np.arctan((l_rear / (l_front + l_rear)) * np.tan(delta))
        state_dot[0] = v * math.cos(theta + beta)
        state_dot[1] = v * math.sin(theta + beta)
        state_dot[2] = (v / l_rear) * math.sin(beta)
        state_dot[3] = a
        state_dot[4] = s_dot

        return state_dot

    def step_nonlinear_model(self, state, input):
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        dt = self.model_params["dt"]

        # STATES
        theta = state[2]
        v = state[3]

        # INPUTS
        delta = input[0]
        a = input[1]
        s_dot = input[2]

        # Update the state vector
        state_dot = np.zeros((5, 1))
        beta = np.arctan((l_rear / (l_front + l_rear)) * np.tan(delta))
        state_dot[0] = v * math.cos(theta + beta)
        state_dot[1] = v * math.sin(theta + beta)
        state_dot[2] = (v / l_rear) * math.sin(beta)
        state_dot[3] = a
        state_dot[4] = s_dot

        new_state = state + state_dot * dt
        return new_state

    def linearization_model(self, state, input, debug=False):

        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]

        # STATES
        theta = state[2]
        v = state[3]
        # INPUTS
        delta = input[0]
        a = input[1]
        s_dot = input[2]

        A = np.zeros((self.n_states, self.n_states))
        B = np.zeros((self.n_states, self.n_inputs))
        alpha = l_rear / (l_front + l_rear)
        beta = np.arctan(alpha * np.tan(delta))

        # Calculate the Jacobian A
        A[0, 2] = -v * math.sin(theta + beta) * self.mpc_params["dt"]
        A[0, 3] = v * math.cos(theta + beta) * self.mpc_params["dt"]

        A[1, 2] = v * math.cos(theta + beta) * self.mpc_params["dt"]
        A[1, 3] = v * math.sin(theta + beta) * self.mpc_params["dt"]

        A[2, 3] = 1 / l_rear * math.sin(beta) * self.mpc_params["dt"]

        for i in range(5):
            A[i, i] = 1

        # Calculate the Jacobian B
        B = np.zeros((self.n_states, 1))
        d_beta_dtan_delta = 1 / (alpha**2 * np.tan(delta) ** 2 + 1)
        d_tan_delta_d_delta = 1 / (math.cos(delta) ** 2) * alpha
        d_beta_d_delta = d_beta_dtan_delta * d_tan_delta_d_delta
        B[0, 0] = -v * math.sin(theta + beta) * d_beta_d_delta * self.mpc_params["dt"]
        B[1, 0] = v * math.cos(theta + beta) * d_beta_d_delta * self.mpc_params["dt"]
        B[2, 0] = (v / l_rear) * math.cos(beta) * d_beta_d_delta * self.mpc_params["dt"]

        B[3, 1] = a * self.mpc_params["dt"]
        B[4, 2] = s_dot * self.mpc_params["dt"]

        # Jacobian C
        C = np.zeros((5, 3))

        if debug:
            print("Linearized State Matrix A:")
            print(A)
            print("\nLinearized Input Matrix B:")
            print(B)
            print("\nMatrix C:")
            print(C)

        return A, B, C, C

    def output_model(self, state, reference, inputs):
        x_ref = reference[0]
        y_ref = reference[1]
        theta_ref = reference[2]
        output = np.zeros((self.model_params["n_outputs"], 1))
        alpha = self.model_params["l_r"] / (
            self.model_params["l_f"] + self.model_params["l_r"]
        )
        delta = inputs[0]
        beta = np.arctan(alpha * np.tan(delta))
        output[0] = math.sin(theta_ref) * (state[0] - x_ref) - math.cos(theta_ref) * (
            state[1] - y_ref
        )
        output[1] = -math.cos(theta_ref) * (state[0] - x_ref) - math.sin(theta_ref) * (
            state[1] - y_ref
        )
        output[2] = inputs[1] * np.sin(beta) + (
            state[3] ** 2 / self.model_params["l_r"]
        ) * np.sin(beta) * np.cos(beta)
        return output


class KinematicBicycleVariableSpeed(MPCModel):
    def nonlinear_model(self, state, input):
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]

        theta = state[2]
        v = state[3]
        delta = input[0]
        a = input[1]

        beta = np.arctan((l_rear / (l_front + l_rear)) * np.tan(delta))
        state_dot = np.zeros((3,))
        x_dot = v * math.cos(theta + beta)
        y_dot = v * math.sin(theta + beta)
        theta_dot = (v / l_rear) * math.sin(beta)
        v_dot = a

        state_dot[0] = x_dot
        state_dot[1] = y_dot
        state_dot[2] = theta_dot
        state_dot[3] = v_dot

        return state_dot
    
    def linearization_model(
        self, state, inputs, reference_state=None, reference_input=None, debug=False
    ):
        """
        Linearizes the vehicle model around the given state and input.

        Args:
            state (list): The current state of the vehicle [x, y, theta].
            input (list): The current input to the vehicle [delta].
            debug (bool): If True, prints the matrices A, B, and C.

        Returns:
            tuple: A tuple containing the linearized state matrix A, the linearized input matrix B, and matrix C.
        """
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        v = self.model_params["v"]
        theta = state[2]
        delta = inputs[0]
        a = inputs[1]

        # Calculate beta
        alpha = l_rear / (l_front + l_rear)
        beta = np.arctan(alpha * np.tan(delta))
        d_beta_dtan_delta = 1 / (alpha**2 * np.tan(delta) ** 2 + 1)
        d_tan_delta_d_delta = 1 / (math.cos(delta) ** 2) * alpha
        d_beta_d_delta = d_beta_dtan_delta * d_tan_delta_d_delta

        # Jacobian A
        A = np.zeros((4, 4))
        for i in range(4):
            A[i, i] = 1
        A[0, 2] = -v * math.sin(theta + beta) * self.mpc_params["dt"]
        A[0, 3] = v * math.cos(theta + beta) * self.mpc_params["dt"]
        A[1, 2] = v * math.cos(theta + beta) * self.mpc_params["dt"]
        A[1, 3] = v * math.sin(theta + beta) * self.mpc_params["dt"]

        # Jacobian B
        B = np.zeros((4, 2))
        B[0, 0] = -v * math.sin(theta + beta) * d_beta_d_delta * self.mpc_params["dt"]
        B[1, 0] = v * math.cos(theta + beta) * d_beta_d_delta * self.mpc_params["dt"]
        B[2, 0] = (v / l_rear) * math.cos(beta) * d_beta_d_delta * self.mpc_params["dt"]
        B[3, 1] = self.mpc_params["dt"]

        # Jacobian C
        C = np.zeros((4, 4))

        if debug:
            print("Linearized State Matrix A:")
            print(A)
            print("\nLinearized Input Matrix B:")
            print(B)
            print("\nMatrix C:")
            print(C)

        return A, B, C, B

    def output_model(self, state, input):
        return state
    
    def step_nonlinear_model(self, state, inp, debug=False):

        new_state = np.zeros((4,))
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        v = self.model_params["v"]
        x = state[0]
        y = state[1]
        theta = state[2]
        v = state[3]
        delta = inp[0]
        a = inp[1]

        beta = np.arctan((l_rear / (l_front + l_rear)) * np.tan(delta))
        x_dot = v * math.cos(theta + beta)
        y_dot = v * math.sin(theta + beta)
        theta_dot = (v / l_rear) * math.sin(beta)

        new_state[0] = x_dot * self.mpc_params["dt"] + x
        new_state[1] = y_dot * self.mpc_params["dt"] + y
        new_state[2] = theta_dot * self.mpc_params["dt"] + theta
        new_state[3] = a * self.mpc_params["dt"] + v
        return new_state
