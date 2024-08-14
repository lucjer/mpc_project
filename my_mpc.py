import osqp as op
import numpy as np
import osqp as op
import scipy as sp
import matplotlib.pyplot as plt
from tqdm import tqdm
from scipy import sparse

import vehicle_models as vm
import mpc_controllers_new as mpc
import mpc_project_new.utils as utils 



# MPC CONTROLLER INITIALIZATION
model_kin = vm.KinematicBicycleConstantSpeed('mpc_bicycle_config.yaml')
N = 5
Q = sparse.diags([1, .2, 1.0])
R = sparse.diags([2.5])
QN = sparse.diags([.1, .1, .1])
InputConstraints = {'umin': np.array([-np.inf]),
                    'umax': np.array([np.inf])}
StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf]),
                    'xmax': np.array([np.inf, np.inf, np.inf])}
mpc_test_2 = mpc.GenericNMPC(model=model_kin, N=N, Q = Q, R=R, QN=Q, StateConstraints=StateConstraints, InputConstraints=InputConstraints) 

# SOLVE NMPC PROBLEM
current_state = np.array([.0, 0.2, 0.])
X_ref = [np.array([1.0 * (i+1), 0., 0.0]) for i in range(N)]
U_ref = [np.zeros((1, )) for i in range(N)]

X_guesss, U_guesss = mpc_test_2.solve_sqp(current_state, X_ref, U_ref, debug = False, sqp_iter=1)
control_input = U_guesss[0]
print(control_input)




if False:
  ############################## MPC MODEL ####################################
  n_states = 3
  n_inputs = 1
  # 
  # model_kin = vm.KinematicBicycleConstantSpeed('mpc_bicycle_config.yaml')
  dt = model_kin.mpc_params['dt']
  #model_kin_sim = vm.KinematicBicycleConstantSpeed('mpc_bicycle_config.yaml')
  #model_kin = vm.KinematicBicycleConstantSpeed('mpc_bicycle_config.yaml')

  ############################ MPC CONTROLLER #################################
  n_horizon = 10
  ############################ TRAJ GENERATION ################################
  x_ref = np.linspace(0,  40 * np.pi, 400)
  y_ref = (np.cos(0.01 * x_ref) * 5 + 2 * np.cos(0.035 * x_ref) * 2) * 2
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
    total_reference_input.append(np.array([rsteer_i]))


  n_total_traj = len(total_reference)
  input_history = []
  current_state = np.array([0.0, 18, -0.5])
  state_history = [current_state]
  n_sim = n_total_traj - n_horizon
  t = 0
  time_steps = []
  print("TOTAL REFERENCE: ", len(total_reference))
  print(total_reference)
  for i in tqdm(range(n_sim)): 
    X = total_reference[i:i+n_horizon]
    U = total_reference_input[i:i+n_horizon]
    print(X)
    input()
    # SETUP MPC
    #X = np.zeros((n_states * n_horizon, 1))
    #U = np.zeros((n_inputs * n_horizon, 1))

    # TRAJECTORY FETCHING
    #X[0] = current_state[0]
    #X[1] = current_state[1]
    #X[2] = current_state[2]
    
    #for k in range(1, n_horizon):
    #  for l in range(n_states):
    #    X[k*n_states + l] = current_traj_states[k][l]
    #for k in range(n_horizon):
    #  for l in range(n_inputs):
    #    U[k*n_inputs + l] = current_traj_inputs[k]
        
    # MPC SETUP
   
    
    X_guess, U_guess = mpc_test_2.solve_sqp(current_state, X, U, debug = False)
    opt_input = U_guess[0]
    current_state_pred = current_state.reshape(n_states, )
    current_state = model_kin.step_nonlinear_model(current_state, [opt_input]).reshape(n_states, 1)
    assert current_state.shape == (n_states, 1), 'Shape of current state is not correct: {}'.format(current_state.shape)
    print(current_state.shape)
                                                                                
    traj_pred_x = [current_state_pred[0]]
    traj_pred_y = [current_state_pred[1]]
    traj_target_x = [X[i][0] for i in range(n_horizon)]  
    traj_target_y = [X[i][1] for i in range(n_horizon)]

    
    for j in range(len(U_guess)):    
      current_state_pred = model_kin.step_nonlinear_model(current_state_pred, [U_guess[j]])
      traj_pred_x.append(current_state_pred[0])
      traj_pred_y.append(current_state_pred[1])
      
    if False:  
      t += dt
      time_steps.append(t)
      plt.figure()
      plt.plot(list(traj_target_x), list(traj_target_y), 'o', label = 'target')
      plt.plot(traj_pred_x, traj_pred_y, 'o', label = 'pred')
      plt.legend()
      plt.savefig('temporary_traj.pdf')
      plt.close()
      if i%10==0:
        input()  

    input_history.append(opt_input)
    state_history.append(current_state)
    
    #plt.plot(current_traj_states)
  #plt.show()
  ############################# FINAL PLOT ##############################
  x_traj = [state_history[i][0] for i in range(n_sim)]
  y_traj = [state_history[i][1] for i in range(n_sim)]
  theta_traj = [state_history[i][2] for i in range(n_sim)]


  plt.plot(x_ref, y_ref, color='blue', label = 'reference')
  print(x_traj)
  print(y_traj)
  input()

  plt.plot(x_traj[2:], y_traj[2:], color='orange', label = 'traj')
  plt.legend()
  plt.savefig('test.pdf')
  plt.show()

  plt.figure()
  plt.plot(time_steps, x_traj)
  plt.savefig('speed.pdf')
