import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import sparse
import vehicles as vm 
import mpc_controllers as mpc
import utils
import os
from scipy.ndimage import gaussian_filter1d



def apply_gaussian_filter(data, sigma):
    """
    Apply Gaussian filter to smooth the data.
    :param data: Input data to be smoothed
    :param sigma: Standard deviation of the Gaussian kernel
    :return: Smoothed data
    """
    return gaussian_filter1d(data, sigma=sigma)

# Exponential Moving Average Function
def exponential_moving_average(data, alpha):
    ema = [data[0]]  # Start the EMA with the first data point
    for i in range(1, len(data)):
        ema.append(alpha * data[i] + (1 - alpha) * ema[-1])
    return ema

# PathFollowerVariableSpeed Class Definition
class PathFollowerVariableSpeed:
    """ Path follower that follows a path with variable speed """
    def __init__(self, path, ds, mpc_controller, velocity_resolution):
        """ Constructor for the PathFollowerVariableSpeed class """
        self.path = path # Collection of (X, Y) points that define the path
        self.default_ds = ds # Default distance between points on the path
        self.current_index = None # Index of the current point on the path
        self.mpc_controller = mpc_controller # MPC controller for the vehicle 
        self.velocity_resolution = velocity_resolution # Resolution of the velocity profile
        self.ds_resolution = self.velocity_resolution * self.mpc_controller.model.mpc_params['dt'] # Distance between points on the path for the velocity profile
        self.trajectory = {'x': [], 'y': [], 'yaw': [], 'v': [], 'inp': [], 'k':[]} # Trajectory of the vehicle
        self.last_X_guess = None # Last predicted state sequence
        self.last_U_guess = None # Last predicted input sequence
        
    def fetch_reference(self, current_state):
        # Find the index of the closest point on the path
        self.current_index = self.find_closest_index(current_state, self.trajectory)
        # Find the index of the next point on the path
        X = []
        U = []
        fetching_index = self.current_index
        for _ in range(self.mpc_controller.N):
            step = int(self.trajectory['v'][fetching_index] // self.velocity_resolution)
            reference_state = np.array([self.trajectory['x'][fetching_index + step],
                                        self.trajectory['y'][fetching_index + step],
                                        self.trajectory['yaw'][fetching_index + step],
                                        self.trajectory['v'][fetching_index + step]])
            fetching_index += step
            X.append(reference_state)
            U.append(self.trajectory['inp'][self.current_index + step])
        return X, np.array(U)
    
    def compute_optimal_input(self, current_state, debug=False):
        X, U = self.fetch_reference(current_state)
        X_guess, U_guess = self.mpc_controller.profile_solve_sqp(current_state, X, U, debug=debug, sqp_iter=3, alpha=0.1)
        return U_guess  
    
    def populate_trajectory(self):
        rx, ry, ryaw, rk, rsteer, s = utils.calc_spline_course(self.path['x'], self.path['y'], ds=self.ds_resolution)
        for rx_i, ry_i, ryaw_i, rsteer_i, rk_i in zip(rx, ry, ryaw, rsteer, rk):
            self.trajectory['x'].append(rx_i)
            self.trajectory['y'].append(ry_i)
            self.trajectory['yaw'].append(ryaw_i)
            a_max = 4
            max_velocity = 15
            ref_velocity = min(a_max /np.abs(rk_i**2*25+1/8.), max_velocity)
            self.trajectory['v'].append(ref_velocity)
            self.trajectory['inp'].append([float(rsteer_i) * 1, 0.])
            self.trajectory['k'].append(rk_i**2)
            
        alpha = 0.0008  # Smoothing factor
        self.trajectory['yaw'] = exponential_moving_average(self.trajectory['yaw'], 0.99)
        self.trajectory['v'] = apply_gaussian_filter(self.trajectory['v'], sigma=2800)
        plt.plot(self.trajectory['v'])
        plt.savefig('simulation_results/velocity_profile.pdf')
        input()
            
    @staticmethod
    def find_closest_index(current_state, trajectory, start_index=0, end_index=None):
        distances = np.sqrt((np.array([x for x in trajectory['x']]) - current_state[0])**2+ 
                            (np.array([y for y in trajectory['y']]) - current_state[1])**2)
        return np.argmin(distances)


    def plot_fetched_trajectory_and_input(self, current_state, optimal_input, n_sim, save_dir='debug_plots'):
        """Plot the fetched reference trajectory and compare the reference input with the optimal input."""
        X_ref, U_ref = self.fetch_reference(current_state)

        ref_x = [state[0] for state in X_ref]
        ref_y = [state[1] for state in X_ref]
        ref_yaw = [state[2] for state in X_ref]
        ref_v = [state[3] for state in X_ref]
        ref_steer = [inp[0] for inp in U_ref]
        ref_acc = [inp[1] for inp in U_ref]

        optimal_steer = [inp[0] for inp in optimal_input]
        optimal_acc = [inp[1] for inp in optimal_input]

        fig, axs = plt.subplots(3, 2, figsize=(15, 12))

        # Plot the fetched reference trajectory (Position)
        axs[0, 0].plot(ref_x, ref_y, 'go-', label='Fetched Reference')
        axs[0, 0].scatter(current_state[0], current_state[1], color='red', label='Current State')
        axs[0, 0].set_title(f"Step {n_sim}: Current Position and Fetched Reference")
        axs[0, 0].set_xlabel("X")
        axs[0, 0].set_ylabel("Y")
        axs[0, 0].legend()
        axs[0, 0].axis('equal')

        # Plot the yaw angle in the fetched reference
        axs[0, 1].plot(ref_yaw, 'm*-', label='Yaw')
        axs[0, 1].set_title("Yaw Angle in Fetched Reference")
        axs[0, 1].set_xlabel("Step")
        axs[0, 1].set_ylabel("Yaw [rad]")
        axs[0, 1].legend()

        # Plot the velocity profile in the fetched reference
        axs[1, 0].plot(ref_v, 'b*-', label='Velocity')
        axs[1, 0].set_title("Velocity Profile in Fetched Reference")
        axs[1, 0].set_xlabel("Step")
        axs[1, 0].set_ylabel("Velocity [m/s]")
        axs[1, 0].legend()

        # Plot the steering input comparison
        axs[1, 1].plot(ref_steer, 'bo-', label='Reference Steering')
        axs[1, 1].plot(optimal_steer, 'ro-', label='Optimal Steering')
        axs[1, 1].set_title("Steering Input Comparison")
        axs[1, 1].set_xlabel("Horizon Step")
        axs[1, 1].set_ylabel("Steering [rad]")
        axs[1, 1].legend()

        # Plot the acceleration input comparison
        axs[2, 0].plot(ref_acc, 'bo-', label='Reference Acceleration')
        axs[2, 0].plot(optimal_acc, 'ro-', label='Optimal Acceleration')
        axs[2, 0].set_title("Acceleration Input Comparison")
        axs[2, 0].set_xlabel("Horizon Step")
        axs[2, 0].set_ylabel("Acceleration [m/s^2]")
        axs[2, 0].legend()

        # Hide the empty subplot (2nd column in the 3rd row)
        axs[2, 1].axis('off')

        plt.tight_layout()

        # Ensure the directory exists
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)

        # Save the figure
        plt.savefig(f'{save_dir}/fetched_trajectory_and_input_step_{n_sim}.pdf')
        plt.close()


# Initialization of the PathFollower and the MPC Controller
x_ref = np.linspace(0, 120 * np.pi, 1200)
y_ref = np.sin(x_ref/40 * np.pi ) * 30 - np.sin(x_ref/20 * np.pi ) * 10
current_state = np.array([0., 5., np.pi/4, 10.0])

model_kin = vm.KinematicBicycleVariableSpeed('config_files/mpc_bicycle_velocity_config.yaml') 
n_horizon = 25
Q = sparse.diags([50, 50, 65, 60])
R = sparse.diags([5, 1])


u_min_acc = -3
u_max_acc = 3
delta_min = -np.pi/6
delta_max = np.pi/6

InputConstraints = {'umin': [delta_min, u_min_acc], 
                    'umax': [delta_max, u_max_acc]}

StateConstraints = {'xmin': np.array([-np.inf, -np.inf, -np.inf, -np.inf]),
                    'xmax': np.array([np.inf, np.inf, np.inf, np.inf])}
mpc_controller = mpc.NMPCSolver(model=model_kin, N=n_horizon, Q=Q, R=R,
                             StateConstraints=StateConstraints,
                             InputConstraints=InputConstraints,
                             debug_plotting_callback=utils.plot_xref_evolution,
                             debug_plots_folder='debug_plots/') 

path_follower = PathFollowerVariableSpeed({'x': x_ref, 'y': y_ref}, 0.05, mpc_controller, 0.05)
path_follower.populate_trajectory()

# INITIALIZE SIMULATION
state_history = [current_state]
input_history = []
path_follower.compute_optimal_input(current_state)

j = path_follower.find_closest_index(current_state, path_follower.trajectory)

n_sim = 0
debug = False


# Initialize lists to store velocities
vehicle_velocities = []
reference_velocities = []
reference_curvatures = []

while n_sim < 25:
    debug = False
    if n_sim % 50 == 0:
        debug = True
        
    noise_std_devs = [0.2, 0.2, 0.1, 0.1]  # Different standard deviations for each component
    noise = np.array([np.random.normal(0, std_dev) for std_dev in noise_std_devs])
    disturbed_state = current_state + noise * 0
    optimal_input = path_follower.compute_optimal_input(disturbed_state, debug=debug)
    current_state = model_kin.step_nonlinear_model(current_state, optimal_input[0])  
    input_history.append(optimal_input[0])
    state_history.append(current_state)
    
    vehicle_velocities.append(current_state[3])

    # Find the closest index on the reference trajectory
    closest_index = path_follower.find_closest_index(current_state, path_follower.trajectory)

    # Record the reference velocity at the closest point
    reference_velocities.append(path_follower.trajectory['v'][closest_index])
    reference_curvatures.append(1/max(0.01, path_follower.trajectory['k'][closest_index]))
    
    # Plot fetched trajectory and input comparison
    if debug:
        path_follower.plot_fetched_trajectory_and_input(current_state, optimal_input, n_sim)
        input()
    
    n_sim += 1



x_traj = [state_history[i][0] for i in range(n_sim)]
y_traj = [state_history[i][1] for i in range(n_sim)]
theta_traj = [state_history[i][2] for i in range(n_sim)]
v_traj = [state_history[i][3] for i in range(n_sim)]


# Final Plot
x_traj = [state_history[i][0] for i in range(2, n_sim)]
y_traj = [state_history[i][1] for i in range(2, n_sim)]
theta_traj = [state_history[i][2] for i in range(2, n_sim)]



plt.legend()

# Save the animation


plt.figure(figsize=(10, 6))
plt.plot(vehicle_velocities, label='Vehicle Velocity', color='blue')
plt.plot(reference_velocities, label='Reference Velocity', color='green', linestyle='--')
plt.title("Vehicle Velocity vs Reference Velocity")
plt.xlabel("Simulation Step")
plt.ylabel("Velocity [m/s]")
plt.legend()
plt.grid(True)
plt.savefig('simulation_results/velocity_comparison.pdf')
plt.show()



delta_input = [inp[0] for inp in input_history]
acc_input = [inp[1] for inp in input_history]
plt.figure(figsize=(10, 6))
plt.subplot(2, 1, 1)
plt.plot(delta_input, label='Steering Input', color='blue')
plt.plot([delta_min]*len(input_history), label='Minimum Steering', color='red', linestyle='dashdot')
plt.plot([delta_max]*len(input_history), label='Maximum Steering', color='red', linestyle='--') 
plt.legend()
plt.subplot(2, 1, 2)
plt.plot(acc_input, label='Acceleration Input', color='green')
plt.plot([u_min_acc]*len(input_history), label='Minimum Accel', color='red', linestyle='dashdot')
plt.plot([u_max_acc]*len(input_history), label='Maximum Accel', color='red', linestyle='--')
plt.savefig('simulation_results/input_comparison.pdf')
plt.legend()
plt.show()




fig, ax = plt.subplots()
ax.plot(x_ref[:n_sim], y_ref[:n_sim], color='blue', label='reference')
line, = ax.plot([], [], color='orange', label='trajectory')

def init():
    line.set_data([], [])
    return line,

def update(frame):
    line.set_data(x_traj[:frame], y_traj[:frame])
    return line,

ani = animation.FuncAnimation(fig, update, frames=n_sim, init_func=init, blit=True)
ani.save('simulation_results/trajectory_animation.gif', writer='imagemagick')



