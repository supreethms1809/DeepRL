from re import A
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.signal import cont2discrete

@dataclass
class simData:
    m: float = 1  # mass
    M: float = 5 # cart mass
    L: float = 2 # length of the pendulum
    g: float = -10 # acceleration due to gravity
    d: float = 1 # damping 
    b: float = 1 # pendulum up fixed

    # Parameters for the simulation
    dt: float = 0.001 # time step
    T: float = 20 # total simulation time
    initial_state: np.ndarray = field(default_factory=lambda: np.array([[2], [1], [2], [1]])) # initial state
    desired_state: np.ndarray = field(default_factory=lambda: np.array([[0], [0], [0], [0]])) # desired state

    # A: np.ndarray = np.array([[0, 1, 0, 0],
    #                           [0, -d/M, (b*m*g)/M, 0],
    #                           [0, 0, 0, 1],
    #                           [0, (-b*d)/M*L, (-b*(M+m))*g/(M * L), 0]])
    # B: np.ndarray = np.array([[0], [1/M], [0], [b/(L*M)]])

    
class Pendulum:
    def __init__(self, data: simData):
        self.data = data
        self.state = data.initial_state
        self.desired_state = data.desired_state
        self.time = 0
        self.history = []
        self.A = np.array([[0, 1, 0, 0],
                              [0, -self.data.d/self.data.M, (self.data.b*self.data.m*self.data.g)/self.data.M, 0],
                              [0, 0, 0, 1],
                              [0, (-self.data.b*self.data.d)/self.data.M*self.data.L, (-self.data.b*(self.data.M+self.data.m))*self.data.g/(self.data.M * self.data.L), 0]])
        self.B = np.array([[0], [1/self.data.M], [0], [self.data.b/(self.data.L*self.data.M)]])
        

    def c2d(self):
        A_d, B_d, _, _, _ = cont2discrete((self.A, self.B,None,None), self.data.dt)
        return A_d, B_d

    def non_linear_dynamics(self, state, action):
        x1, x2, x3, x4 = state.flatten()
        x,v,theta,w = state.flatten()
        m, M, L, g, d, b = self.data.m, self.data.M, self.data.L, self.data.g, self.data.d, self.data.b
        u = action.flatten()
        dx1 = dx = v
        dx2 = dv = ((-(m**2)*(L**2)*g*np.cos(theta)*np.sin(theta)) + 
                    (m*(L**2)*((m*L*(w**2))*np.sin(theta)-(d*v))) + 
                    (m*(L**2)*u)) / (m*(L**2)*(M+m*(1-(np.cos(theta)**2))))
        dx3 = dtheta = w 
        dx4 = dw = (((m+M)*m*g*L*np.sin(theta)) - 
                    (m*L*np.cos(theta)*(m*L*(w**2)*np.sin(theta)-(d*v))) + 
                    (m*L*np.cos(theta)*u))/(m*(L**2)*(M+m*(1-(np.cos(theta)**2))))
        return np.array([[dx1], [dx2.item()], [dx3], [dx4.item()]])
    
    def lin_dynamics(self, state, action):
        return self.A @ state + self.B @ action
    
    def simulate(self):
        # Simulate the pendulum dynamics
        time_steps = int(self.data.T / self.data.dt)
        for t in range(time_steps):
            #print(f"Timestep t: {t} Time: {self.time:.2f}")
            action = np.array([[0]])  # Example action
            self.state = self.dynamics(pendulum.state, action)
            self.history.append(self.state.flatten())
            self.time += self.data.dt
        self.plot_results()
    
    def reset(self):
        self.state = self.data.initial_state
        self.time = 0
        self.history = []

    def plot_results(self):
        # Plot the results
        history = np.array(self.history)
        plt.figure(figsize=(10, 5))
        plt.plot(history[:, 0], label='x (position)')
        plt.plot(history[:, 1], label='v (velocity)')
        plt.plot(history[:, 2], label='theta (angle)')
        plt.plot(history[:, 3], label='w (angular velocity)')
        plt.xlabel('Time steps')
        plt.ylabel('State values')
        plt.title('Pendulum Dynamics Simulation')
        plt.legend()
        plt.grid()
        plt.show()

    def LQRBackwardRecursion(self):
        pass

    def LQRForwardPassRecursion(self):
        pass


if __name__ == "__main__":
    data = simData()
    pendulum = Pendulum(data)
    A_d, B_d = pendulum.c2d()
    debug_log = False
    if debug_log:
        print("Pendulum parameters:", data)
        print("Discrete A matrix:", A_d)
        print("Discrete B matrix:", B_d)
        print("Initial state:", pendulum.state)
        print("Desired state:", pendulum.desired_state)
        print("Dynamics:", pendulum.dynamics(pendulum.state, np.array([[0]])))  # Example action
        print("Pendulum simulation complete.")

    pendulum.LQRBackwardRecursion()
    pendulum.LQRForwardPassRecursion()
    
    pendulum.simulate()  # Run the simulation