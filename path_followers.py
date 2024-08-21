import utils
import numpy as np
import mpc_controllers as mpc
from scipy import sparse
import vehicle_models as vm
import matplotlib.pyplot as plt
import matplotlib.animation as animation


def exponential_moving_average(data, alpha):
    ema = [data[0]]  # Start the EMA with the first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema


class PathFollowerVariableSpeed:
    """ Path follower that follows a path with variable speed """
    def __init__(self, path, ds, mpc_controller, velocity_resolution):
        """ Constructor for the PathFollowerVariableSpeed class """
        self.path = path # Collection of (X, Y) points that define the path
        self.default_ds = ds # Default distance between points on the path
        self.current_index = None # Index of the current point on the path
        self.mpc_controller = mpc_controller # MPC controller for the vehicle 
        self.velocity_resolution = 0.5 # Resolution of the velocity profile
        self.ds_resolution = self.velocity_resolution * self.mpc_controller.model.mpc_params['dt'] # Distance between points on the path for the velocity profile
        self.trajectory = {'x': [], 'y': [], 'yaw': [], 'v': [], 'inp': []} # Trajectory of the vehicle
        
    def fetch_reference(self, current_state):
        # Find the index of the closest point on the path
        self.current_index = utils.find_closest_index(current_state, self.trajectory)
        # Find the index of the next point on the path
        X = []
        U = []
        for _ in range(self.mpc_controller.N):
            step = int(self.trajectory['v'][self.current_index] // self.velocity_resolution)

            reference_state = np.array([self.trajectory['x'][self.current_index + step],
                                        self.trajectory['y'][self.current_index + step],
                                        self.trajectory['yaw'][self.current_index + step],
                                        self.trajectory['v'][self.current_index + step]])
            X.append(reference_state)
            U.append(self.trajectory['inp'][self.current_index + step])
        return X, U
    
    def compute_optimal_input(self, current_state, debug = False):
        X, U = self.fetch_reference(current_state)
        X_guess, U_guess = self.mpc_controller.solve_sqp(current_state, X, U, debug = debug, sqp_iter=15, alpha=0.1)
        return U_guess[0]
    
    
    def populate_trajectory(self):
        rx, ry, ryaw, rk, rsteer, s = utils.calc_spline_course(self.path['x'], self.path['y'], ds=self.ds_resolution)
        for rx_i, ry_i, ryaw_i, rsteer_i, rk_i in zip(rx, ry, ryaw, rsteer, rk):
            self.trajectory['x'].append(rx_i)
            self.trajectory['y'].append(ry_i)
            self.trajectory['yaw'].append(ryaw_i)
            # TODO: Naive implementation of velocity profile
            a_max = 9.8
            max_velocity = 9.8
            ref_velocity = min(a_max /np.abs(np.abs(rk_i)*2+0.05), max_velocity)
            self.trajectory['v'].append(ref_velocity)
            self.trajectory['inp'].append([rsteer_i, 0])
            
        # Smoothen the yaw and v vectors
        alpha = 0.1  # Smoothing factor
        self.trajectory['yaw'] = exponential_moving_average(self.trajectory['yaw'], alpha)
        self.trajectory['v'] = exponential_moving_average(self.trajectory['v'], alpha)
    
    @staticmethod
    def find_closest_index(current_state, trajectory):
        print("Trajectory")
        print(trajectory['x'][0])
        input()
        distances = np.sqrt((np.array([x for x in trajectory['x']]) - current_state[0])**2+ 
                            (np.array([y for y in trajectory['y']]) - current_state[1])**2)
        return np.argmin(distances)
    
    
x_ref = np.linspace(0,  10 * np.pi, 400) * 2
y_ref = (np.cos(0.2 * x_ref) * 6 + 2 * np.cos(0.25 * x_ref) * 2) * 2
model_kin = vm.KinematicBicycleVariableSpeed('config_files/mpc_bicycle_velocity_config.yaml') 
n_horizon = 20
Q = sparse.diags([10, 10, 200, 15])
R = sparse.diags([1, 20])
QN = sparse.diags([.1, .1, .1, .1])
# TODO: Implementation of input and state constraints handling
InputConstraints = {'umin': np.array([-np.inf, -np.inf]), 
                    'umax': np.array([np.inf, np.inf])}
StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                    'xmax': np.array([np.inf, np.inf, np.inf, np.inf])}
mpc_controller = mpc.NMPCSolver(model=model_kin, N=n_horizon, Q = Q, R=R, QN=Q,
                             StateConstraints=StateConstraints,
                             InputConstraints=InputConstraints,
                             alpha = 0.5) 

path_follower = PathFollowerVariableSpeed({'x': x_ref, 'y': y_ref}, 0.5, mpc_controller, 0.1)
path_follower.populate_trajectory()


# INITIALIZE SIMULATION
current_state = np.array([0, 20, 0., 7])
state_history = [current_state]
input_history = []
path_follower.compute_optimal_input(current_state)


j = path_follower.find_closest_index(current_state, path_follower.trajectory)
n_sim = 0

while n_sim <200:
    debug = False
    if n_sim % 50 == 0:
        debug = True
        print(n_sim)
        input()
    optimal_input = path_follower.compute_optimal_input(current_state, debug = False)
    current_state = model_kin.step_nonlinear_model(current_state, optimal_input).reshape(model_kin.n_states, 1)
    input_history.append(optimal_input)
    state_history.append(current_state)
    n_sim += 1


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