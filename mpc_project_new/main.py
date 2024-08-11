import osqp as op
import numpy as np
import osqp as op
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

import vehicle_models as vm
import mpc_controllers as mpc


model_kin = vm.KinematicBicycleConstantSpeed('mpc_bicycle_config.yaml')
mpc_test_2 = mpc.GenericNMPC(model=model_kin, Q=np.eye(3), R=np.eye(2), QN=np.eye(3), StateConstraints={}, InputConstraints={}) 