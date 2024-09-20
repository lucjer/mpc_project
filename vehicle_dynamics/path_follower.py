
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation
from scipy import sparse
import vehicle_dynamics.vehicle_models as vm 
import spline 
import os
from scipy.ndimage import gaussian_filter1d


class PathFollower:
    """ Path follower that follows a path with variable speed """
    def __init__(self, path, mpc_controller,  ds=0.1, velocity_resolution=0.1, maximum_velocity=15, maximum_lateral_acceleration=3, search_radius=500, sigma=1000):
        """ Constructor for the PathFollowerVariableSpeed class """
        self.path = path # Collection of (X, Y) points that define the path
        self.default_ds = ds # Default distance between points on the path
        self.current_index = None # Index of the current point on the path
        self.mpc_controller = mpc_controller # MPC controller for the vehicle 
        self.velocity_resolution = velocity_resolution # Resolution of the velocity profile
        self.maximum_lateral_acceleration = maximum_lateral_acceleration
        self.maximum_velocity = maximum_velocity
        self.ds_resolution = self.velocity_resolution * self.mpc_controller.model.mpc_params['dt'] # Distance between points on the path for the velocity profile
        self.trajectory = {'x': [], 'y': [], 'yaw': [], 'v': [], 'inp': [], 'k':[]} # Trajectory of the vehicle
        self.search_radius = search_radius # Search radius for finding the closest point on the path
        self.sigma = sigma # Sigma - Smoothening window width for velocity profile
        self.stopping = False
        
    def fetch_reference(self, current_state):
        # Find the index of the closest point on the path
        self.current_index = self.find_closest_index(current_state, self.trajectory, self.current_index, self.search_radius)
        # Find the index of the next point on the path
        X = []
        U = []
        fetching_index = self.current_index
        complete_fetch = True
        for _ in range(self.mpc_controller.N):
            step = int(self.trajectory['v'][fetching_index] // self.velocity_resolution)
            index_to_fetch = fetching_index + step
            if index_to_fetch < len(self.trajectory['x']): 
                reference_state = np.array([self.trajectory['x'][index_to_fetch],
                                            self.trajectory['y'][index_to_fetch],
                                            self.trajectory['yaw'][index_to_fetch],
                                            self.trajectory['v'][index_to_fetch]])
            else: # If out of range, toggle the car stop
                complete_fetch = False
                break
            X.append(reference_state)
            U.append(self.trajectory['inp'][index_to_fetch])
            fetching_index += step

        if complete_fetch:
            stopping = False
            return X, np.array(U), stopping
        else:
            stopping = True
            return None, None, stopping
    
    def compute_optimal_input(self, current_state, debug=False):
        if not self.stopping:
            X, U, self.stopping = self.fetch_reference(current_state)
        if not self.stopping: # If the end of the path is not reached
            X_guess, U_guess = self.mpc_controller.profile_solve_sqp(current_state, X, U, debug=debug)
        else: # Naively stop the car ignoring the deviation from the actual path.
            X = np.zeros((self.mpc_controller.N, self.mpc_controller.nx))
            U = np.zeros((self.mpc_controller.N, self.mpc_controller.nu))
            self.mpc_controller.Q  = sparse.diags([0, 0, 0, 20]) # State cost matrix - Only control velocity to stop the car
            X_guess, U_guess= self.mpc_controller.profile_solve_sqp(current_state, X, U, debug=debug)
        return U_guess  
    
    def populate_trajectory(self):
        rx, ry, ryaw, rk, rsteer, s = spline.calc_spline_course(self.path['x'], self.path['y'], ds=self.ds_resolution)
        for rx_i, ry_i, ryaw_i, rsteer_i, rk_i in zip(rx, ry, ryaw, rsteer, rk):
            self.trajectory['x'].append(rx_i)
            self.trajectory['y'].append(ry_i)
            self.trajectory['yaw'].append(ryaw_i)
            ref_velocity = min(np.sqrt(self.maximum_lateral_acceleration / np.abs(rk_i + 0.01)), self.maximum_velocity)
            self.trajectory['v'].append(ref_velocity)
            # TODO: Add the acceleration input to the trajectory. 
            # For now, set it to 0, optimization can handle it and minimize acceleration inputs (deviation from 0)
            self.trajectory['inp'].append([float(rsteer_i), 0.])
            self.trajectory['k'].append(np.abs(rk_i))
        self.trajectory['v'] = gaussian_filter1d(self.trajectory['v'], sigma=self.sigma)


    def find_closest_index(self, current_state, trajectory, start_index=0, search_radius=500):
        if search_radius is None: # If no search radius is provided, search the entire trajectory
            distances = np.sqrt((np.array([x for x in trajectory['x']]) - current_state[0])**2+ 
                            (np.array([y for y in trajectory['y']]) - current_state[1])**2)
            return np.argmin(distances)
        
        # Determine the search range
        if start_index is None:
            start = 0
            end = len(trajectory['x'])
        else:
            start = max(start_index - search_radius, 0)
            end = min(start_index + search_radius, len(trajectory['x']))
            
        # Compute distances within the search range - No need to search whole trajectory
        distances = np.sqrt((np.array([x for x in trajectory['x'][start:end]]) - current_state[0])**2+ 
                            (np.array([y for y in trajectory['y'][start:end]]) - current_state[1])**2)
        
        # Find the index of the closest point within the search range
        min_index_in_window = np.argmin(distances)
        
        # Return the absolute index
        return start + min_index_in_window


    def plot_fetched_trajectory_and_input(self, current_state, optimal_input, n_sim, save_dir='debug_plots'):
        """Used for debug and demonstration, plot the fetched reference trajectory and compare the reference input with the optimal input."""
        X_ref, U_ref, _ = self.fetch_reference(current_state)

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


