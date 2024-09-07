import numpy as np
import math

from mpc_model import MPCModel
from numba import jit

class KinematicBicycleVariableSpeed(MPCModel):

    def nonlinear_model(self, state, input):
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]

        return self._nonlinear_model_jit(state, input, l_front, l_rear)

    def linearization_model(self, state, inputs, reference_state=None, reference_input=None, debug=False):
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        dt = self.mpc_params["dt"]

        A, B, C = self._linearization_model_jit(state, inputs, l_front, l_rear, dt)

        if debug:
            print("Linearized State Matrix A:")
            print(A)
            print("\nLinearized Input Matrix B:")
            print(B)
            print("\nMatrix C:")
            print(C)

        return A, B, C, B

    def step_nonlinear_model(self, state, inp, debug=False):
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        dt = self.mpc_params["dt"]

        if debug:
            return state

        return self._step_nonlinear_model_jit(state, inp, l_front, l_rear, dt)

    @staticmethod
    @jit(nopython=True)
    def _nonlinear_model_jit(state, input, l_front, l_rear):
        theta = state[2]
        v = state[3]
        delta = input[0]
        a = input[1]

        beta = math.atan((l_rear / (l_front + l_rear)) * math.tan(delta))
        state_dot = np.zeros((4,))
        x_dot = v * math.cos(theta + beta)
        y_dot = v * math.sin(theta + beta)
        theta_dot = (v / l_rear) * math.sin(beta)
        v_dot = a

        state_dot[0] = x_dot
        state_dot[1] = y_dot
        state_dot[2] = theta_dot
        state_dot[3] = v_dot

        return state_dot

    @staticmethod
    @jit(nopython=True)
    def _linearization_model_jit(state, inputs, l_front, l_rear, dt):
        theta = state[2]
        v = state[3]
        delta = inputs[0]
        a = inputs[1]

        # Calculate beta
        alpha = l_rear / (l_front + l_rear)
        beta = math.atan(alpha * math.tan(delta))
        d_beta_dtan_delta = 1 / (alpha**2 * math.tan(delta) ** 2 + 1)
        d_tan_delta_d_delta = 1 / (math.cos(delta) ** 2) * alpha
        d_beta_d_delta = d_beta_dtan_delta * d_tan_delta_d_delta

        # Jacobian A
        A = np.zeros((4, 4))
        A[0, 0] = 1
        A[1, 1] = 1
        A[2, 2] = 1
        A[3, 3] = 1
        A[0, 2] = -v * math.sin(theta + beta) * dt
        A[0, 3] = math.cos(theta + beta) * dt
        A[1, 2] = v * math.cos(theta + beta) * dt
        A[1, 3] = math.sin(theta + beta) * dt

        # Jacobian B
        B = np.zeros((4, 2))
        B[0, 0] = -v * math.sin(theta + beta) * d_beta_d_delta * dt
        B[1, 0] = v * math.cos(theta + beta) * d_beta_d_delta * dt
        B[2, 0] = (v / l_rear) * math.cos(beta) * d_beta_d_delta * dt
        B[3, 1] = dt

        # Jacobian C
        C = np.zeros((4, 4))

        return A, B, C

    @staticmethod
    @jit(nopython=True)
    def _step_nonlinear_model_jit(state, inp, l_front, l_rear, dt):
        new_state = np.zeros((4,))
        theta = state[2]
        v = state[3]
        delta = inp[0]
        a = inp[1]

        beta = math.atan((l_rear / (l_front + l_rear)) * math.tan(delta))
        x_dot = v * math.cos(theta + beta)
        y_dot = v * math.sin(theta + beta)
        theta_dot = (v / l_rear) * math.sin(beta)

        new_state[0] = x_dot * dt + state[0]
        new_state[1] = y_dot * dt + state[1]
        new_state[2] = theta_dot * dt + theta
        new_state[3] = a * dt + v

        return new_state

    def output_model(self, state, input):
        return state

class KinematicBicycleConstantSpeed(MPCModel):
    def __init__(self, model_params, mpc_params):
        self.model_params = model_params
        self.mpc_params = mpc_params

    def nonlinear_model(self, state, input):

        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        v = self.model_params["v"]

        return self._nonlinear_model_jit(state, input, l_front, l_rear, v)

    def linearization_model(self, state, input, reference_state=None, reference_input=None, debug=False):
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        v = self.model_params["v"]
        dt = self.mpc_params["dt"]

        A, B, C = self._linearization_model_jit(state, input, l_front, l_rear, v, dt)

        if debug:
            print("Linearized State Matrix A:")
            print(A)
            print("\nLinearized Input Matrix B:")
            print(B)
            print("\nMatrix C:")
            print(C)

        return A, B, C, B

    def step_nonlinear_model(self, state, input, debug=False):
        l_front = self.model_params["l_f"]
        l_rear = self.model_params["l_r"]
        v = self.model_params["v"]
        dt = self.mpc_params["dt"]

        if debug:
            return state

        return self._step_nonlinear_model_jit(state, input, l_front, l_rear, v, dt)

    @staticmethod
    @jit(nopython=True)
    def _nonlinear_model_jit(state, input, l_front, l_rear, v):
        theta = state[2]
        delta = input[0]
        beta = math.atan((l_rear / (l_front + l_rear)) * math.tan(delta))
        state_dot = np.zeros((3,))
        x_dot = v * math.cos(theta + beta)
        y_dot = v * math.sin(theta + beta)
        theta_dot = (v / l_rear) * math.sin(beta)

        state_dot[0] = x_dot
        state_dot[1] = y_dot
        state_dot[2] = theta_dot

        return state_dot

    @staticmethod
    @jit(nopython=True)
    def _linearization_model_jit(state, input, l_front, l_rear, v, dt):
        theta = state[2]
        delta = input[0]

        # Calculate beta
        alpha = l_rear / (l_front + l_rear)
        beta = math.atan(alpha * math.tan(delta))
        d_beta_dtan_delta = 1 / (alpha**2 * math.tan(delta) ** 2 + 1)
        d_tan_delta_d_delta = 1 / (math.cos(delta) ** 2) * alpha
        d_beta_d_delta = d_beta_dtan_delta * d_tan_delta_d_delta

        # Jacobian A
        A = np.zeros((3, 3))
        A[0, 0] = 1
        A[1, 1] = 1
        A[2, 2] = 1
        A[0, 2] = -v * math.sin(theta + beta) * dt
        A[1, 2] = v * math.cos(theta + beta) * dt

        # Jacobian B
        B = np.zeros((3, 1))
        B[0, 0] = -v * math.sin(theta + beta) * d_beta_d_delta * dt
        B[1, 0] = v * math.cos(theta + beta) * d_beta_d_delta * dt
        B[2, 0] = (v / l_rear) * math.cos(beta) * d_beta_d_delta * dt

        # Jacobian C
        C = np.zeros((3, 3))

        return A, B, C

    @staticmethod
    @jit(nopython=True)
    def _step_nonlinear_model_jit(state, input, l_front, l_rear, v, dt):
        new_state = np.zeros((3,))
        theta = state[2]
        delta = input[0]
        beta = math.atan((l_rear / (l_front + l_rear)) * math.tan(delta))
        x_dot = v * math.cos(theta + beta)
        y_dot = v * math.sin(theta + beta)
        theta_dot = (v / l_rear) * math.sin(beta)

        new_state[0] = x_dot * dt + state[0]
        new_state[1] = y_dot * dt + state[1]
        new_state[2] = theta_dot * dt + theta
        return new_state

    def output_model(self, state, input):
        return state
