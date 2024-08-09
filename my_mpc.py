import osqp as op
import numpy as np
import osqp as op
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm

import vehicle_models as vm
import mpc_controllers as mpc
import utils 


############################## MPC MODEL ####################################
dt = 0.08     # Discretization interval
n_states = 3
n_inputs = 1
model_kin = vm.UnicycleConstantSpeed('mpc_unicycle_config.yaml')
model_kin_sim = vm.KinematicBicycleConstantSpeed('mpc_bicycle_config.yaml')
############################ MPC CONTROLLER #################################
n_horizon = 30
Q = np.diag([0.01, 10, 10]) * 5
R = np.eye(n_inputs) * 10
S = np.eye(n_inputs) * 10
mpc_kin = mpc.MPCController(model_kin, Q, R, S, n_horizon)
############################ TRAJ GENERATION ################################
x_ref = np.linspace(0,  20 * np.pi, 400)
y_ref = np.cos(0.02 * x_ref) * 1 + 2 * np.cos(0.075 * x_ref) * 3
v = 1 # TODO: DO NOT CHANGE - Need additional changes in the code. Fine for now
ds = v * dt # s is the distance along the path (x_ref, y_ref) - ds the distance interval
# I just used an open source library to calculate all the needed variables along trajectories
# I splined to obtain the yaw and the curvature. The reference steering is compute according to the curvature
# k = curvature, l = wheelbase, d = steering angle -> k * l = tan(d) -> d = arctan(k * l)
rx, ry, ryaw, rk, rsteer, s = utils.calc_spline_course(x_ref, y_ref, ds=ds)
total_reference = []
total_reference_input = []
#TODO: The next section is just needed to have dimensions match properly. Not optimal but fine for now
for rx_i, ry_i, ryaw_i, rsteer_i in zip(rx, ry, ryaw, rsteer):
  total_reference.append(np.array([rx_i, ry_i, ryaw_i]))
  total_reference_input.append(rsteer_i)


n_total_traj = len(total_reference)
input_history = []
current_state = np.array([-0.1, 7.0, -0.00])
state_history = [current_state]
n_sim = n_total_traj - n_horizon
t = 0
time_steps = []
for i in tqdm(range(n_sim)): #
  # fetch the traj used by mpc - atm the first linearization point is not the current state. ignore for now
  current_traj_states = total_reference[i:i+n_horizon]
  current_traj_inputs = total_reference_input[i:i+n_horizon]
  #for j in range(n_horizon):
  #  x_flattened[i * n_states:(i+1) * n_states, 0] = current_traj_states[j]
  #  u_flattened[i * n_inputs:(i+1) * n_states, 0] = current_traj_inputs[j]
  # SETUP MPC
  X = np.ones((n_states * n_horizon, 1))
  U = np.ones((n_inputs * n_horizon, 1))

  X[0] = current_state[0]
  X[1] = current_state[1]
  X[2] = current_state[2]
  for k in range(1, n_horizon):
    for l in range(n_states):
      X[k*n_states + l] = current_traj_states[k][l]
  for k in range(n_horizon):
    for l in range(n_inputs):
      U[k*n_inputs + l] = current_traj_inputs[k]

  G_x, L_x = mpc_kin.get_gamma_lambda_x(X, U, n_horizon)
  H, f = mpc_kin.build_hessian_f(Q, R, S, L_x, G_x, X, n_horizon)
  A = np.eye((n_inputs * n_horizon))
  l = - np.ones((n_inputs * n_horizon)) * np.pi/4
  u = + np.ones((n_inputs * n_horizon)) * np.pi/4
  H_sparse = sp.sparse.csc_matrix(H)
  A_sparse = sp.sparse.csc_matrix(A)
  m = op.OSQP()
  m.setup(P=H_sparse, q=f.transpose(), A=A_sparse, l=l, u=u, verbose=False)
  results = m.solve()
  opt_input = results.x[0]
  if False:
    print()
    print("SIMULATION: ", i)
    print("Fetched state: ", X)
    print("Wish it was: ", X)
    print("Fetched input: ", U)
    print("G_X: ")
    print(G_x)
    print("L_X: ")
    print(L_x)
    print("H: ")
    print(H)
    print("f: ")
    print(f)
    print("LOL")
    print("Optimal input: ", opt_input)

  
  current_state = model_kin_sim.nonlinear_model(current_state, [opt_input]) * dt * 1 + current_state
  t += dt
  time_steps.append(t)

  
  #print("Optimal input: ", opt_input)
  input_history.append(opt_input)
  state_history.append(current_state)
  
  #plt.plot(current_traj_states)
#plt.show()
############################# FINAL PLOT ##############################
x_traj = [state_history[i][0] for i in range(n_sim)]
y_traj = [state_history[i][1] for i in range(n_sim)]
theta_traj = [state_history[i][2] for i in range(n_sim)]


plt.plot(x_ref, y_ref, color='blue', label = 'reference')
plt.plot(x_traj[1:], y_traj[1:], color='orange', label = 'traj')
plt.legend()
plt.savefig('test.pdf')
plt.show()

plt.figure()
plt.plot(time_steps, x_traj)
plt.savefig('speed.pdf')
