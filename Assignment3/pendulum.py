import re
import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from pyparsing import C
from regex import F
from scipy.signal import cont2discrete
import sympy as sp

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
        self.initial_state = data.initial_state
        self.desired_state = data.desired_state
        self.state = None
        self.action = None
        self.desired_action = np.zeros((1, 1))
        self.time = 0
        self.time_steps = int(self.data.T / self.data.dt)
        self.history = []
        self.A = np.array([[0, 1, 0, 0],
                              [0, -self.data.d/self.data.M, (self.data.b*self.data.m*self.data.g)/self.data.M, 0],
                              [0, 0, 0, 1],
                              [0, (-self.data.b*self.data.d)/self.data.M*self.data.L, (-self.data.b*(self.data.M+self.data.m))*self.data.g/(self.data.M * self.data.L), 0]])
        self.B = np.array([[0], [1/self.data.M], [0], [self.data.b/(self.data.L*self.data.M)]])
        self.Ad, self.Bd = self.c2d()
        self.Q  = np.eye(4)  # State cost matrix
        self.R = 0.01 * np.eye(1)

        self.output = {
            'time': [],
            'Q_t': [],
            'q_t': [],
            'Qofx_t_u_t': [],
            'u_t': [None] * self.time_steps,
            'K_t': [],
            'k_t': [],
            'V_t': [],
            'v_t': [],
            'Vofx_t': []
        }

        

    # convert continuous-time system to discrete-time system
    def c2d(self):
        A_d, B_d, _, _, _ = cont2discrete((self.A, self.B,None,None), self.data.dt)
        return A_d, B_d

    # Dynamics function - non-linear
    def non_linear_dynamics(self, state, action):
        x1, x2, x3, x4 = np.array(state).flatten()
        x,v,theta,w = np.array(state).flatten()
        m, M, L, g, d, b = self.data.m, self.data.M, self.data.L, self.data.g, self.data.d, self.data.b
        u = np.array(action).flatten()
        dx1 = dx = v
        dx2 = dv = ((-(m**2)*(L**2)*g* np.cos(theta) * np.sin(theta)) + 
                    (m*(L**2)*((m*L*(w**2))* np.sin(theta)-(d*v))) + 
                    (m*(L**2)*u)) / (m*(L**2)*(M+m*(1-(np.cos(theta)**2))))
        dx3 = dtheta = w 
        dx4 = dw = (((m+M)*m*g*L* np.sin(theta)) - 
                    (m*L* np.cos(theta)*(m*L*(w**2)* np.sin(theta)-(d*v))) + 
                    (m*L* np.cos(theta)*u))/(m*(L**2)*(M+m*(1-(np.cos(theta)**2))))
        return sp.Matrix([[dx1], [dx2.item()], [dx3], [dx4.item()]])
    
    # Dynamics function - linear
    def lin_dynamics(self, state, action):
        return self.Ad @ state + self.Bd @ action
    
    # simulate the pendulum dynamics with either linear or non-linear dynamics
    def simulate(self, dynamics="linear", output=None):
        if dynamics == "linear":
            self.dynamics = self.lin_dynamics
        elif dynamics == "non-linear":
            self.dynamics = self.non_linear_dynamics
        else:
            raise ValueError("Dynamics must be 'linear' or 'non-linear'")
        
        self.reset()  
        # Simulate the pendulum dynamics
        self.state = self.data.initial_state
        self.time = 0
        self.history.append(np.array(self.state).flatten())
        for t in range(self.time_steps):
            self.action = output['u_t'][t]
            self.next_state = self.dynamics(self.state, self.action)
            self.history.append(np.array(self.state).flatten())
            self.state = self.next_state
            self.time += self.data.dt
        self.plot_results()
    
    # Reset the simulation to the initial state
    def reset(self):
        self.state = self.data.initial_state
        self.time = 0
        self.history = []

    # Plot the results of the simulation
    def plot_results(self):
        # Plot the results
        history = np.array(self.history)
        #print("History shape:", history.shape)
        #print("History:", history)
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

    def LQRBackwardForwardRecursion(self):
        x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
        u = sp.symbols('u')
        state = sp.Matrix([x1, x2, x3, x4]) - sp.Matrix(self.desired_state)
        action = sp.Matrix([u])
        
        A = sp.Matrix(self.Ad)
        B = sp.Matrix(self.Bd)
        Q = sp.Matrix(self.Q)
        R = sp.Matrix(self.R)

        f_x_u =  A @ state + B @ action
        # Cost function J = sigma (x_t^T Q x_t + u_t^T R u_t)
        cost_func = state.T @ Q @ state +  action @ R @ action

        variables = state.col_join(action)
        # F_t = del x_t f(x_t,u_t), f_t = del u_t f(x_t,u_t)
        F_t = f_x_u.jacobian(variables)
        f_t = F_t[:, -1]
        
        # C_t = doubledel x_t cost_func, c_t = del u_t cost_func
        c_t = sp.Matrix([sp.diff(cost_func[0], var) for var in variables])
        C_t = sp.hessian(cost_func, variables)

        self.state = self.initial_state
        self.action = self.desired_action

        # Delta x_t and delta u_t
        delta_x_t = self.state - self.desired_state
        delta_u_t = self.action - self.desired_action

        # Define V_t+1 and v_t+1
        V_t_plus_1 = np.zeros((4, 4))
        v_t_plus_1 = np.zeros((4, 1))

        # Run the backward recursion
        print("Backward Recursion")
        for t in range(self.time_steps - 1, -1, -1):
            self.output['time'].append(self.time)

            # Calculate Q_t,q_t
            Q_t = C_t + F_t.T @ V_t_plus_1 @ F_t
            q_t = c_t + F_t.T @ V_t_plus_1 @ f_t + F_t.T @ v_t_plus_1

            # Calculate Q(x_t,u_t)
            vecstate = sp.Matrix(self.state - self.desired_state)
            vecaction = sp.Matrix(self.action - self.desired_action)
            stacked_vec = vecstate.col_join(vecaction)
            Qofx_t_u_t = 0.5 * (stacked_vec).T @ Q_t @ (stacked_vec) + stacked_vec.T @ q_t

            # Calculate u_t
            #u_t = np.argmin(Qofx_t_u_t)
            #self.action = u_t

            # Calculate Quu and Qux, Qxu and Qxx
            Quu = Q_t[4:5, 4:5]
            Qux = Q_t[4:5, 0:4]
            Qxu = Q_t[0:4, 4:5]
            Qxx = Q_t[0:4, 0:4]
            qx = sp.Matrix(q_t[0:4])
            qu = sp.Matrix(q_t[4:5])

            # Calculate K_t and k_t]
            K_t = -sp.Inverse(Quu) @ Qux
            k_t = -sp.Inverse(Quu) @ qu

            # Calculate V_t and v_t
            V_t = Qxx + Qxu @ K_t + K_t.T @ Qux + K_t.T @ Quu @ K_t
            v_t = qx + Qxu @ k_t + K_t.T @ qu + K_t.T @ Quu @ k_t

            # Calculate Vofx_t 
            Vofx_t = 0.5 * self.state.T @ V_t @ self.state + self.state.T @ v_t

            self.time -= self.data.dt

            # Store the results
            self.output['Q_t'].append(Q_t)
            self.output['q_t'].append(q_t)
            self.output['Qofx_t_u_t'].append(Qofx_t_u_t)
            #self.output['u_t'].append("")
            self.output['K_t'].append(K_t)
            self.output['k_t'].append(k_t)
            self.output['V_t'].append(V_t)
            self.output['v_t'].append(v_t)
            self.output['Vofx_t'].append(Vofx_t)
        
        print("Forward Recursion")
        # Run the forward recursion
        self.state = self.desired_state
        for t in range(self.time_steps):
            u_t = self.output['K_t'][t] @ self.state + self.output['k_t'][t]
            self.action = u_t
            self.output['u_t'][t] = u_t
            self.time += self.data.dt
            self.history.append(self.state.flatten())

        return self.output, self.history

if __name__ == "__main__":
    data = simData()
    pendulum = Pendulum(data)
    debug_log = False
    if debug_log:
        print("Pendulum parameters:", data)
        print("Initial state:", pendulum.state)
        print("Desired state:", pendulum.desired_state)
        print("Dynamics:", pendulum.dynamics(pendulum.state, np.array([[0]])))  # Example action
        print("Pendulum simulation complete.")

    output, history = pendulum.LQRBackwardForwardRecursion()
    
    # # Print the output
    # print("Output:")
    # for key, value in output.items():
    #     if key == 'k_t':
    #         print(f"{key}: {value}")
    #         exit()

    # Simulate the pendulum dynamics
    pendulum.simulate(dynamics="linear", output=output)
    #pendulum.simulate(dynamics="non-linear", output=output)