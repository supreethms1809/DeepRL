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
    initial_state: np.ndarray = field(default_factory=lambda: np.array([[2], [0], [2], [0]])) # initial state
    desired_state: np.ndarray = field(default_factory=lambda: np.array([[0], [0], [0], [0]])) # desired state

    # A: np.ndarray = np.array([[0, 1, 0, 0],
    #                           [0, -d/M, (b*m*g)/M, 0],
    #                           [0, 0, 0, 1],
    #                           [0, (-b*d)/M*L, (-b*(M+m))*g/(M * L), 0]])
    # B: np.ndarray = np.array([[0], [1/M], [0], [b/(L*M)]])

    
class Pendulum:
    def __init__(self, data: simData):
        # Data given
        self.data = data
        self.initial_state = data.initial_state
        self.desired_state = data.desired_state

        # local variables
        self.current_state = None
        self.current_action = None
        self.next_state = None
        self.next_action = None
        self.start_action = np.zeros((1, 1))
        self.time = 0
        self.time_steps = int(self.data.T / self.data.dt)

        # A and B matrices for continious time system
        self.A = np.array([[0, 1, 0, 0],
                              [0, -self.data.d/self.data.M, (self.data.b*self.data.m*self.data.g)/self.data.M, 0],
                              [0, 0, 0, 1],
                              [0, (-self.data.b*self.data.d)/self.data.M*self.data.L, (-self.data.b*(self.data.M+self.data.m))*self.data.g/(self.data.M * self.data.L), 0]])
        self.B = np.array([[0], [1/self.data.M], [0], [self.data.b/(self.data.L*self.data.M)]])

        # A and B for discrete time system
        self.Ad, self.Bd = self.c2d()

        # State cost matrix
        self.Q  = np.eye(4)  # State cost matrix
        self.R = 0.01 * np.eye(1)

        # Outputs
        self.output = []
        self.outputwithaction = []
        self.trajectory = []

    # convert continuous-time system to discrete-time system
    def c2d(self):
        Ad, Bd, _, _, _ = cont2discrete((self.A, self.B,None,None), self.data.dt)
        return Ad, Bd

    # Dynamics function - non-linear
    def non_linear_dynamics(self, state, action):
        x1, x2, x3, x4 = np.array(state).flatten()
        x, v, theta, w = np.array(state).flatten()
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
        self.current_state = self.data.initial_state
        self.current_action = self.start_action
        self.time = 0.0

        for t in range(0, self.time, self.time_steps):
            # Save time and state
            self.trajectory['step'].append(t)
            self.trajectory['time'].append(self.time)
            
            # Get the current state and action
            self.current_action = output['u_t'][t]
            self.next_state = self.dynamics(self.current_state, self.current_action)

            # Store the trajectory
            self.trajectory['state'].append(np.array(self.current_state, dtype = np.float32).flatten())
            self.trajectory['action'].append(np.array(self.current_action, dtype = np.float32).flatten())
            self.trajectory['next_state'].append(np.array(self.next_state, dtype = np.float32).flatten())
            
            self.current_state = self.next_state
            self.time += self.data.dt
        
        return self.trajectory
    
    # Reset the simulation to the initial state
    def reset(self):
        self.state = self.data.initial_state
        self.time = 0.0

    # Plot the results of the simulation
    def plot_results(self):
        # Plot the results
        history = np.array(self.outputwithaction)
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
        # Define the symbolic variables
        x1, x2, x3, x4 = sp.symbols('x1 x2 x3 x4')
        u1 = sp.symbols('u1')
        state = sp.Matrix([x1, x2, x3, x4])
        action = sp.Matrix([u1])
        
        A = sp.Matrix(self.Ad)
        B = sp.Matrix(self.Bd)
        Q = sp.Matrix(self.Q)
        R = sp.Matrix(self.R)

        f_x_u =  A @ state + B @ action
        # Cost function J = sigma (x_t^T Q x_t + u_t^T R u_t)
        cost_func = state.T @ Q @ state +  action.T @ R @ action

        variables = state.col_join(action)
        # F_t = del x_t f(x_t,u_t), f_t = del u_t f(x_t,u_t)
        F_t_dir = f_x_u.jacobian(state)
        fl_t_dir = f_x_u.jacobian(action)
        
        # C_t = doubledel x_t cost_func, c_t = del u_t cost_func
        cl_t_dir = cost_func.jacobian(variables)
        C_t_dir = sp.hessian(cost_func, variables)

        initial_state = np.array(self.data.initial_state).flatten()
        initial_action = np.array(self.start_action).flatten()
        subs_dict = {x1: initial_state[0], x2: initial_state[1]
                     , x3: initial_state[2], x4: initial_state[3]
                     , u1: initial_action[0]}

        F_t_numeric = F_t_dir.subs(subs_dict).evalf()
        F_t = np.array(F_t_numeric, dtype=np.float64)

        fl_t_numeric = fl_t_dir.subs(subs_dict).evalf()
        fl_t = np.array(fl_t_numeric, dtype=np.float64)

        C_t_numeric = C_t_dir.subs(subs_dict).evalf()
        C_t = np.array(C_t_numeric, dtype=np.float64)

        cl_t_numeric = cl_t_dir.subs(subs_dict).evalf()
        cl_t = np.array(cl_t_numeric, dtype=np.float64)

        print(f"C_t shape: {C_t}")
        print(f"c_t shape: {cl_t}")
        # F_t = F_t_numpy[:, :4]
        # f_t = F_t_numpy[:, 4:]
        # C_t = C_t_numpy[:4, :4]
        # c_t = c_t_numpy[:4]

        # set the initial state and action
        self.current_state = self.initial_state
        self.current_action = self.start_action

        # Define V_t+1 and v_t+1
        V_t_plus_1 = np.zeros((4, 4))
        v_t_plus_1 = np.zeros((4, 1))

        # Run the backward recursion
        print("Backward Recursion")
        self.time = self.data.T
        for t in range(self.time_steps, -1, -1):
            # Calculate Q_t,q_t
            Q_t = C_t + F_t.T @ V_t_plus_1 @ F_t
            q_t = cl_t + F_t.T @ V_t_plus_1 @ fl_t + F_t.T @ v_t_plus_1

            # Calculate Q(x_t,u_t)
            vecstate = self.current_state
            Qofx_t_u_t = 0.5 * (vecstate).T @ Q_t @ (vecstate) + vecstate.T @ q_t

            # Calculate u_t
            u_t = np.argmin(Qofx_t_u_t)
            self.next_action = np.array([[u_t]], dtype=np.float64)

            # Calculate Quu and Qux, Qxu and Qxx
            print("Q_t shape:", Q_t.shape)
            print("Q_t:", Q_t.__class__)
            Quu = Q_t[4:5, 4:5]
            Qux = Q_t[4:5, 0:4]
            Qxu = Q_t[0:4, 4:5]
            Qxx = Q_t[0:4, 0:4]
            qx = q_t[0:4, 0]
            qu = q_t[4:5, 0]

            # Calculate K_t and k_t]
            K_t = (-np.linalg.inv(Quu) @ Qux)
            k_t = (-np.linalg.inv(Quu) @ qu)

            # Calculate V_t and v_t
            V_t = Qxx + Qxu @ K_t + K_t.T @ Qux + K_t.T @ Quu @ K_t
            v_t = qx + Qxu @ k_t + K_t.T @ qu + K_t.T @ Quu @ k_t

            # Calculate Vofx_t 
            Vofx_t = 0.5 * self.current_state.T @ V_t @ self.current_state + self.current_state.T @ v_t

            self.next_state = self.lin_dynamics(self.current_state, self.current_action)
            
            # Store the results
            output = {
                'time': t,
                'state': self.current_state,
                'K_t': K_t,
                'k_t': k_t,
            }

            self.output.insert(0,output)

            self.current_state = self.next_state
            self.current_action = self.next_action
            self.time -= self.data.dt
        
        print("Forward Recursion")
        # Run the forward recursion
        self.state = self.desired_state
        self.time = 0
        for t in range(self.time_steps):
            matching_elements = [entry for entry in self.output if entry['time'] == self.time]
            print("Matching elements:", matching_elements)
            if matching_elements:
                K_t = matching_elements[0]['K_t']
                k_t = matching_elements[0]['k_t']
                state = matching_elements[0]['state']
            else:
                raise ValueError("No matching elements found in output")
            u_t = K_t @ state + k_t
            action = u_t
            print("Action:", u_t)
            self.state = self.lin_dynamics(state, action)
            # Store the results
            output = {
                'time': t,
                'state': self.current_state,
                'K_t': K_t,
                'k_t': k_t,
                'u_t': u_t,
            }

            self.time += self.data.dt
            self.outputwithaction.append(output)

        return self.output 

if __name__ == "__main__":
    data = simData()
    pendulum = Pendulum(data)
    output, history = pendulum.LQRBackwardForwardRecursion()

    # Simulate the pendulum dynamics
    trajectory = pendulum.simulate(dynamics="linear", output=output)
    with open("pendulum_simulation.txt", "w") as f:
        for key, value in trajectory.items():
            f.write(f"{key}: {value}\n")

    #pendulum.simulate(dynamics="non-linear", output=output)