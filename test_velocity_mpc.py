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
model_kin = vm.KinematicBicycleVariableSpeed('config_files/mpc_bicycle_velocity_config.yaml') 
n_horizon = 30
Q = sparse.diags([1, 1, 100, 1])
R = sparse.diags([1, 100])
QN = sparse.diags([.1, .1, .1, .1])
# TODO: Implementation of input and state constraints handling
InputConstraints = {'umin': np.array([-np.inf, -np.inf]), 
                    'umax': np.array([np.inf, np.inf])}
StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                    'xmax': np.array([np.inf, np.inf, np.inf, np.inf])}
mpc_test_2 = mpc.NMPCSolver(model=model_kin, N=n_horizon, Q = Q, R=R, QN=Q,
                             StateConstraints=StateConstraints,
                             InputConstraints=InputConstraints,
                             alpha = 0.5) 


# SOLVE SAMPLE NMPC PROBLEM
current_state = np.array([.0, 3.5, 0., 0.2])
X_ref = [np.array([1.0 * (i+1) * model_kin.mpc_params['dt'], 0., 0.0, 1]) for i in range(n_horizon)]
U_ref = [np.zeros((2, )) for i in range(n_horizon)]

X_guesss, U_guesss = mpc_test_2.solve_sqp(current_state, X_ref, U_ref,
                                          X_guess = [current_state] * n_horizon, 
                                          U_guess = [np.zeros((2, ))] * n_horizon,
                                          debug = True, sqp_iter=10, alpha=0.5)


# POPULATE REFERENCE TRAJECTORY
x_ref = np.linspace(0,  10 * np.pi, 400)
y_ref = (np.cos(0.5 * x_ref) * 6 + 2 * np.cos(0.75 * x_ref) * 2) * 2
total_reference, total_reference_input = utils.compute_reference_velocity_test(model_kin.mpc_params['dt'], x_ref, y_ref)
n_total_traj = len(total_reference)


# INITIALIZE SIMULATION
current_state = np.array([0, 20, 0.5, 0.4])
state_history = [current_state]
input_history = []
initial_index = utils.find_start_index(current_state, n_horizon, total_reference, total_reference_input)
j = initial_index
n_sim = 0 

while j < n_total_traj - n_horizon:
    n_sim += 1
    j = utils.find_start_index(current_state, n_horizon, total_reference, total_reference_input)
    X = total_reference[j:j+n_horizon]
    U = total_reference_input[j:j+n_horizon]
    if n_sim%1000 == 1:
        X_guess, U_guess = mpc_test_2.solve_sqp(current_state, X, U, debug = True, sqp_iter=5, alpha=0.3)
        input()
    else:
        X_guess, U_guess = mpc_test_2.solve_sqp(current_state, X, U, debug = False, sqp_iter=5, alpha=0.3)

    opt_input = U_guess[0]
    current_state = model_kin.step_nonlinear_model(current_state, opt_input).reshape(model_kin.n_states, 1)
    input_history.append(opt_input)
    state_history.append(current_state)




# PLOTTING STUFF # 

x_traj = [state_history[i][0] for i in range(n_sim)]
y_traj = [state_history[i][1] for i in range(n_sim)]
theta_traj = [state_history[i][2] for i in range(n_sim)]
v_traj = [state_history[i][3] for i in range(n_sim)]

plt.figure()
plt.plot(x_ref, y_ref, color='blue', label = 'reference')
plt.plot(x_traj[1:], y_traj[1:], color='orange', label = 'traj')
plt.legend()
plt.savefig('test.pdf')
plt.show()

plt.figure()
plt.plot(v_traj[1:])
plt.savefig('test_velocity.pdf')



import matplotlib.animation as animation

# Final Plot
x_traj = [state_history[i][0] for i in range(2, n_sim)]
y_traj = [state_history[i][1] for i in range(2, n_sim)]
theta_traj = [state_history[i][2] for i in range(2, n_sim)]

fig, ax = plt.subplots()
ax.plot(x_ref, y_ref, color='blue', label='reference')
ax.axis('equal')
line, = ax.plot([], [], color='orange', label='trajectory')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x_traj[:frame], y_traj[:frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=n_sim, init_func=init, blit=True)

plt.legend()

# Save the animation
ani.save('trajectory_animation.gif', writer='imagemagick')


plt.figure()
plt.plot(input_history)
plt.savefig('input_history.pdf')