import utils
import numpy as np
import mpc_controllers as mpc
from scipy import sparse
import vehicle_models as vm
from scipy.signal import savgol_filter
import matplotlib.pyplot as plt


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
            step = self.trajectory['v'] / self.velocity_resolution 
            reference_state = np.array([self.trajectory['x'][self.current_index + step],
                                        self.trajectory['y'][self.current_index + step],
                                        self.trajectory['yaw'][self.current_index + step],
                                        self.trajectory['v'][self.current_index + step]])
            X.append(reference_state)
            U.append(np.array([self.trajectory['inp'][self.current_index + step]]))
        return X, U
    
    def compute_optimal_input(self, current_state):
        X, U = self.fetch_reference(current_state)
        X_guess, U_guess = self.mpc_controller.solve_sqp(current_state, X, U, debug = False, sqp_iter=5, alpha=0.3)
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
        self.trajectory['yaw'] = savgol_filter(self.trajectory['yaw'], window_length=25, polyorder=5)
        self.trajectory['v'] = savgol_filter(self.trajectory['v'], window_length=25, polyorder=5)
    
    @staticmethod
    def find_closest_index(current_state, trajectory):
        distances = np.sqrt((np.array([point[0] for point in trajectory['x']]) - current_state[0])**2+ 
                            (np.array([point[1] for point in trajectory['y']]) - current_state[1])**2)
        return np.argmin(distances)
    
    
x_ref = np.cos(np.linspace(0, 1000 * np.pi, 100))
y_ref = np.sin(np.linspace(0, 1000 * np.pi, 100)) 
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
mpc_controller = mpc.NMPCSolver(model=model_kin, N=n_horizon, Q = Q, R=R, QN=Q,
                             StateConstraints=StateConstraints,
                             InputConstraints=InputConstraints,
                             alpha = 0.5) 

path_follower = PathFollowerVariableSpeed({'x': x_ref, 'y': y_ref}, 0.5, mpc_controller, 0.1)
path_follower.populate_trajectory()

steer_profile = [inp_i[0] for inp_i in path_follower.trajectory['inp']]
v_profile = path_follower.trajectory['v']

plt.figure(figsize=(10, 5))
plt.plot(v_profile)
plt.savefig('Steer_test.pdf')