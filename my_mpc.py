import osqp as op
import numpy as np
import osqp as op
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse

import vehicle_models as vm
import mpc_controllers as mpc
import utils as utils 



# MPC CONTROLLER INITIALIZATION
model_kin = vm.KinematicBicycleConstantSpeed('mpc_bicycle_config.yaml')
n_horizon = 5
Q = sparse.diags([1, 1, .1])
R = sparse.diags([2.5])
QN = sparse.diags([.1, .1, .1])
InputConstraints = {'umin': np.array([-np.inf]),
                    'umax': np.array([np.inf])}
StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                    'xmax': np.array([np.inf, np.inf, np.inf])}
mpc_test_2 = mpc.GenericNMPC(model=model_kin, N=n_horizon, Q = Q, R=R, QN=Q,
                             StateConstraints=StateConstraints,
                             InputConstraints=InputConstraints,
                             alpha = 0.5) 

# SOLVE SAMPLE NMPC PROBLEM
current_state = np.array([.0, 1.5, 0.])
X_ref = [np.array([1.0 * (i+1) * model_kin.mpc_params['dt'], 0., 0.0]) for i in range(n_horizon)]
U_ref = [np.zeros((1, )) for i in range(n_horizon)]

X_guesss, U_guesss = mpc_test_2.solve_sqp(current_state, X_ref, U_ref, debug = False, sqp_iter=20)
control_input = U_guesss[0]
print(control_input)


# POPULATE REFERENCE TRAJECTORY
dt = model_kin.mpc_params['dt']
x_ref = np.linspace(0,  40 * np.pi, 400)
y_ref = (np.cos(0.2 * x_ref) * 6 + 2 * np.cos(0.35 * x_ref) * 2) * 2
v = 1 
ds = v * dt 
rx, ry, ryaw, rk, rsteer, s = utils.calc_spline_course(x_ref, y_ref, ds=ds)
total_reference = []
total_reference_input = []
for rx_i, ry_i, ryaw_i, rsteer_i in zip(rx, ry, ryaw, rsteer):
  total_reference.append(np.array([rx_i, ry_i, ryaw_i]))
  total_reference_input.append(np.array([rsteer_i]))
  n_total_traj = len(total_reference)
  input_history = []
  current_state = np.array([0.0, 20, 0.5])
  state_history = [current_state]
  n_sim = n_total_traj - n_horizon



t = 0
time_steps = []
for i in tqdm(range(n_sim)): 
  
  X = total_reference[i:i+n_horizon]
  U = total_reference_input[i:i+n_horizon]

  X_guess, U_guess = mpc_test_2.solve_sqp(current_state, X, U, debug = False, sqp_iter=5, alpha=0.2)
  opt_input = U_guess[0]
  current_state_pred = current_state.reshape(model_kin.n_states, )
  current_state = model_kin.step_nonlinear_model(current_state, [opt_input]).reshape(model_kin.n_states, 1)
  input_history.append(opt_input)
  state_history.append(current_state)

#plt.plot(current_traj_states)
#plt.show()
############################# FINAL PLOT ##############################
x_traj = [state_history[i][0] for i in range(n_sim)]
y_traj = [state_history[i][1] for i in range(n_sim)]
theta_traj = [state_history[i][2] for i in range(n_sim)]

plt.figure()
plt.plot(x_ref, y_ref, color='blue', label = 'reference')
plt.plot(x_traj[1:], y_traj[1:], color='orange', label = 'traj')
plt.legend()
plt.savefig('test.pdf')
plt.show()

plt.figure()
plt.plot(time_steps, x_traj)
plt.savefig('speed.pdf')
