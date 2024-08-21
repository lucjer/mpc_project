
import math
import numpy as np
import bisect
import matplotlib.pyplot as plt


class Spline:
    """
    Cubic Spline class
    """

    def __init__(self, x, y):
        self.b, self.c, self.d, self.w = [], [], [], []

        self.x = x
        self.y = y

        self.nx = len(x)  # dimension of x
        h = np.diff(x)

        # calc coefficient c
        self.a = [iy for iy in y]

        # calc coefficient c
        A = self.__calc_A(h)
        B = self.__calc_B(h)
        self.c = np.linalg.solve(A, B)
        #  print(self.c1)

        # calc spline coefficient b and d
        for i in range(self.nx - 1):
            self.d.append((self.c[i + 1] - self.c[i]) / (3.0 * h[i]))
            tb = (self.a[i + 1] - self.a[i]) / h[i] - h[i] * \
                (self.c[i + 1] + 2.0 * self.c[i]) / 3.0
            self.b.append(tb)

    def calc(self, t):
        """
        Calc position
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.a[i] + self.b[i] * dx + \
            self.c[i] * dx ** 2.0 + self.d[i] * dx ** 3.0

        return result

    def calcd(self, t):
        """
        Calc first derivative
        if t is outside of the input x, return None
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = self.b[i] + 2.0 * self.c[i] * dx + 3.0 * self.d[i] * dx ** 2.0
        return result

    def calcdd(self, t):
        """
        Calc second derivative
        """

        if t < self.x[0]:
            return None
        elif t > self.x[-1]:
            return None

        i = self.__search_index(t)
        dx = t - self.x[i]
        result = 2.0 * self.c[i] + 6.0 * self.d[i] * dx
        return result

    def __search_index(self, x):
        """
        search data segment index
        """
        return bisect.bisect(self.x, x) - 1

    def __calc_A(self, h):
        """
        calc matrix A for spline coefficient c
        """
        A = np.zeros((self.nx, self.nx))
        A[0, 0] = 1.0
        for i in range(self.nx - 1):
            if i != (self.nx - 2):
                A[i + 1, i + 1] = 2.0 * (h[i] + h[i + 1])
            A[i + 1, i] = h[i]
            A[i, i + 1] = h[i]

        A[0, 1] = 0.0
        A[self.nx - 1, self.nx - 2] = 0.0
        A[self.nx - 1, self.nx - 1] = 1.0
        #  print(A)
        return A

    def __calc_B(self, h):
        """
        calc matrix B for spline coefficient c
        """
        B = np.zeros(self.nx)
        for i in range(self.nx - 2):
            B[i + 1] = 3.0 * (self.a[i + 2] - self.a[i + 1]) / \
                h[i + 1] - 3.0 * (self.a[i + 1] - self.a[i]) / h[i]
        return B


class Spline2D:
    """
    2D Cubic Spline class
    """

    def __init__(self, x, y):
        self.s = self.__calc_s(x, y)
        self.sx = Spline(self.s, x)
        self.sy = Spline(self.s, y)

    def __calc_s(self, x, y):
        dx = np.diff(x)
        dy = np.diff(y)
        self.ds = np.hypot(dx, dy)
        s = [0]
        s.extend(np.cumsum(self.ds))
        return s

    def calc_position(self, s):
        """
        calc position
        """
        x = self.sx.calc(s)
        y = self.sy.calc(s)

        return x, y

    def calc_curvature(self, s):
        """
        calc curvature
        """
        dx = self.sx.calcd(s)
        ddx = self.sx.calcdd(s)
        dy = self.sy.calcd(s)
        ddy = self.sy.calcdd(s)
        k = (ddy * dx - ddx * dy) / ((dx ** 2 + dy ** 2)**(3 / 2))
        return k

    def calc_yaw(self, s):
        """
        calc yaw
        """
        dx = self.sx.calcd(s)
        dy = self.sy.calcd(s)
        yaw = math.atan2(dy, dx)
        return yaw


def calc_spline_course(x, y, ds=0.1):
    # TODO: Add dependency with respect to the car wheelbase
    sp = Spline2D(x, y)
    s = list(np.arange(0, sp.s[-1], ds))

    rx, ry, ryaw, rk = [], [], [], []
    rsteer = []
    # k * l = tan(delta)
    # delta = arctan(k * l)
    for i_s in s:
        ix, iy = sp.calc_position(i_s)
        rx.append(ix)
        ry.append(iy)
        ryaw.append(sp.calc_yaw(i_s))
        k = sp.calc_curvature(i_s)
        rk.append(k)
        rsteer.append(np.arctan(k))
    return rx, ry, ryaw, rk, rsteer, s


def compute_reference(dt, x_ref, y_ref, v = 1):
    ds = v * dt 
    rx, ry, ryaw, rk, rsteer, s = calc_spline_course(x_ref, y_ref, ds=ds)
    total_reference = []
    total_reference_input = []
    for rx_i, ry_i, ryaw_i, rsteer_i in zip(rx, ry, ryaw, rsteer):
      total_reference.append(np.array([rx_i, ry_i, ryaw_i]))
      total_reference_input.append(np.array([rsteer_i]))
    return total_reference,total_reference_input

def compute_reference_velocity_test(dt, x_ref, y_ref, v = 1):
    ds = v * dt 
    rx, ry, ryaw, rk, rsteer, s = calc_spline_course(x_ref, y_ref, ds=ds)
    total_reference = []
    total_reference_input = []
    for rx_i, ry_i, ryaw_i, rsteer_i in zip(rx, ry, ryaw, rsteer):
      total_reference.append(np.array([rx_i, ry_i, ryaw_i, v]))
      total_reference_input.append(np.array([rsteer_i, 0]))
    return total_reference,total_reference_input



def plot_xref_evolution(X_ref_evolution, filename="xref_evolution.pdf"):
    """
    Plots the evolution of X_ref over the iterations of the SQP algorithm,
    including yaw angles on the trajectories.
    
    Args:
        X_ref_evolution (list of lists): A list containing X_ref at each iteration.
        filename (str): The filename where the plot will be saved.
    """
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
    
    plt.title('Trajectory Evolution with Yaw Angles')
    plt.xlabel('X Position')
    plt.ylabel('Y Position')
    plt.axis('equal')
    plt.legend()
    
    plt.tight_layout()
    
    # Save the plot as a PDF file
    plt.savefig(filename)
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
        
        