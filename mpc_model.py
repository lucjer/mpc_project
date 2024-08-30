import yaml
import math
import abc


class MPCModel(metaclass=abc.ABCMeta):
    """ Abstract class for MPC models. """
    def __init__(self, model_config_yaml):
        with open(model_config_yaml, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f)
        self.model_params = data["model_params"]
        self.mpc_params = data["mpc_params"]
        self.n_states = self.model_params["n_states"]
        self.n_inputs = self.model_params["n_inputs"]

    def nonlinear_model(self, state, inp):
        """Nonlinear vector field. Returns x_dot = f(x, u)."""

    @abc.abstractmethod
    def linearization_model(self, state, inp, reference_state, reference_input):
        """Linearized model around the given state, input, reference"""

    @abc.abstractmethod
    def step_nonlinear_model(self, state, inp):
        """Discrete-time nonlinear model. Returns x_(i+1) given x_i and u_i."""
