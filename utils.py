
import math
import numpy as np
import bisect
import matplotlib.pyplot as plt


def plot_xref_evolution(current_state, results, folder_path):
    """
    Plots the evolution of X_ref over the iterations of the SQP algorithm,
    including yaw angles on the trajectories.
    
    Args:
        X_ref_evolution (list of lists): A list containing X_ref at each iteration.
        filename (str): The filename where the plot will be saved.
    """
    X_ref_evolution = results['X_guess_evolution']
    N = len(X_ref_evolution)
    
    plt.figure(figsize=(10, 5))
    
    # Plot X_ref evolution trajectory with yaw angles
    for i in range(N):
        x_positions = [state[0] for state in X_ref_evolution[i]]
        y_positions = [state[1] for state in X_ref_evolution[i]]
        yaw_angles = [state[2] for state in X_ref_evolution[i]]
        
        # Plot trajectory with yaw angles
        plt.quiver(x_positions, y_positions, np.cos(yaw_angles), np.sin(yaw_angles), angles='xy', scale_units='xy', scale=4, alpha=0.6)
        plt.scatter(x_positions, y_positions, label=f'Iteration {i+1}', alpha=0.6)

    plt.quiver(current_state[0], current_state[1], np.cos(current_state[2]), np.sin(current_state[2]), angles='xy', scale_units='xy', scale=4, alpha=0.6)
    plt.scatter(current_state[0], current_state[1], label=f'Current state', alpha=0.8, marker='d')
    
    plt.title('Trajectory Evolution with Yaw Angles')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot as a PDF file
    plt.savefig(folder_path + 'xref_evolution.pdf')
    plt.close()

def find_closest_index(current_state, total_reference):
    distances = np.sqrt((np.array([point[0] for point in total_reference]) - current_state[0])**2+ 
                        (np.array([point[1] for point in total_reference]) - current_state[1])**2)
    return np.argmin(distances)



def find_start_index(current_state, n_horizon, total_reference, total_reference_input):
    start_index =  find_closest_index(current_state, total_reference)
    # Shift the index forward by one
    start_index = min(start_index + 1, len(total_reference) - n_horizon)
    return start_index


def plot_cost_evolution(cost_evolution, filename="cost_evolution.pdf"):
        plt.figure(figsize=(10, 5))
        plt.plot(cost_evolution)
        plt.savefig(filename)
        
        
def get_triangle_vertices(center, size, theta):
    # Basic triangle pointing up
    vertices = np.array([[0, 1], [-0.6, -0.5], [0.6, -0.5]])
    
    # Scale the triangle to the desired size
    vertices *= size
    
    # Rotate the triangle by theta
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)], 
                                [np.sin(theta),  np.cos(theta)]])
    rotated_vertices = vertices.dot(rotation_matrix)
    
    # Translate the triangle to the center position
    rotated_vertices[:, 0] += center[0]
    rotated_vertices[:, 1] += center[1]
    
    return rotated_vertices