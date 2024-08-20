import numpy as np
import matplotlib.pyplot as plt
import vehicle_models as vm


spatial_kin = vm.KinematicBicycleSpatialSpeed('config_files/mpc_bicycle_spatial_config.yaml')
X = np.linspace(0, 10, 20)
Y = np.zeros_like(X)
plt.plot(X, Y, 'o')
vehicle_pose = np.array([0, 2, 0, 0, 0])
plt.plot(vehicle_pose[0], vehicle_pose[1], 'ro')
plt.savefig('test_spatial.png')

reference_pose = np.array([1, 0, 0, 0, 0])
error = spatial_kin.output_model(vehicle_pose, reference_pose, inputs = [0, 0, 0])
print(error)