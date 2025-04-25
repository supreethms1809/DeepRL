'''
Python script to simulate a pendulum using LQR control.
Run the script to simulate the pendulum dynamics and plot the results.
run cmd:
    python pendulum.py
'''

import numpy as np
import matplotlib.pyplot as plt
from dataclasses import dataclass, field
from scipy.signal import cont2discrete
import sympy as sp
import scipy.linalg

# Pendulum simulation using LQR control given data 
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

# Pendulum class
# This class simulates the pendulum dynamics and implements LQR control
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

        # A and B matrices for NonLinear time system
        self.A = np.array([[0, 1, 0, 0],
                              [0, -self.data.d/self.data.M, (self.data.b*self.data.m*self.data.g)/self.data.M, 0],
                              [0, 0, 0, 1],
                              [0, (-self.data.b*self.data.d)/self.data.M*self.data.L, (-self.data.b*(self.data.M+self.data.m))*self.data.g/(self.data.M * self.data.L), 0]])
        self.B = np.array([[0], [1/self.data.M], [0], [self.data.b/(self.data.L*self.data.M)]])

        # A and B for Linear time system
        self.Al = np.array([[0, 1, 0, 0],
                              [0, -self.data.d/self.data.M, (self.data.b*self.data.m*self.data.g)/self.data.M, 0],
                              [0, 0, 0, 1],
                              [0, (-self.data.b*self.data.d)/self.data.M*self.data.L, (-self.data.b*(self.data.M+self.data.m))*self.data.g/(self.data.M * self.data.L), 0]])
        self.Bl = np.array([[0], [1/self.data.M], [0], [self.data.b/(self.data.L*self.data.M)]])
        # Discrete-time system
        self.Ad, self.Bd = self.c2d()

        # State cost matrix
        self.Q  = np.eye(4)
        self.R = 0.01 * np.eye(1)

        # Outputs
        self.output = []
        self.outputwithaction = []
        self.trajectory = []

    # convert continuous-time system to discrete-time system using scipy library
    def c2d(self):
        Ad, Bd, _, _, _ = cont2discrete((self.Al, self.Bl, None, None), self.data.dt)
        return Ad, Bd

    # Dynamics function - non-linear
    def non_linear_dynamics(self, state, action):
        #x1, x2, x3, x4 = np.array(state).flatten()
        x, v, theta, w = np.array(state).flatten()
        m, M, L, g, d, b = self.data.m, self.data.M, self.data.L, self.data.g, self.data.d, self.data.b
        u = np.array(action).flatten()[0]
        dx1 = dx = v
        dx2 = dv = ((-(m**2)*(L**2)*g* np.cos(theta) * np.sin(theta)) + 
                    (m*(L**2)*((m*L*(w**2))* np.sin(theta)-(d*v))) + 
                    (m*(L**2)*u)) / (m*(L**2)*(M+m*(1-(np.cos(theta)**2))))
        dx3 = dtheta = w 
        dx4 = dw = (((m+M)*m*g*L* np.sin(theta)) - 
                    (m*L* np.cos(theta)*(m*L*(w**2)* np.sin(theta)-(d*v))) + 
                    (m*L* np.cos(theta)*u))/(m*(L**2)*(M+m*(1-(np.cos(theta)**2))))
        return np.array([[dx1], [dx2], [dx3], [dx4]])
    
    # Dynamics function - linear
    def lin_dynamics(self, state, action):
        return self.Ad @ state + self.Bd @ action
    
    # simulate the pendulum dynamics with either linear or non-linear dynamics
    def simulate(self, output=None):
        self.reset()
        # Simulate the pendulum dynamics
        current_state = self.data.initial_state
        current_action = self.start_action
        self.time = 0.0
        
        # Simulate the pendulum dynamics
        for t in range(self.time_steps):
            # Get the current state and action
            current_action = output[t]['u_t']
            next_state = self.lin_dynamics(current_state, current_action)

            # Store the trajectory for plotting
            self.trajectory.append({
                'time': self.time,
                'state': current_state,
                'action': current_action,
                'next_state': next_state
            })

            current_state = next_state
            self.time += self.data.dt
        
        return self.trajectory
    
    # Reset the simulation to the initial state
    def reset(self):
        self.current_state = self.data.initial_state
        self.time = 0.0

    # Plot the results of the simulation
    def plot_simulation_results(self, trajectory):
        import matplotlib.pyplot as plt
        times = [step['time'] for step in trajectory]
        states = np.array([step['state'].flatten() for step in trajectory])
        actions = np.array([step['action'].flatten() for step in trajectory])
        # Clip actions for better visualization
        #actions = np.clip(actions, -2e2, 2e2)
        plt.figure(figsize=(5, 5))
        # Plot state evolution: x, v, theta, omega
        plt.plot(times, states[:, 0], label='x (position)')
        plt.plot(times, states[:, 1], label='v (velocity)')
        plt.plot(times, states[:, 2], label='theta (angle)')
        plt.plot(times, states[:, 3], label='omega (angular velocity)')
        plt.plot(times, actions[:, 0], label='u (control input)', linewidth=2.5)
        plt.xlabel('Time (s)')
        plt.ylabel('State and Action values')
        plt.title('Pendulum Simulation Results')
        plt.legend()
        plt.grid(True)
        plt.show()

    # Plot the results of the simulation state and action separately
    def plot_simulation_results_separately(self, trajectory):
        import matplotlib.pyplot as plt
        times = [step['time'] for step in trajectory]
        states = np.array([step['state'].flatten() for step in trajectory])
        actions = np.array([step['action'].flatten() for step in trajectory])
        
        # Plot state evolution: x, v, theta, omega
        fig, axs = plt.subplots(5, 1, figsize=(10, 10))
        plt.title('Pendulum State Evolution')
        axs[0].plot(times, states[:, 0], label='x (position)')
        axs[0].set_ylabel('x (position)')
        axs[0].set_xlabel('Time (s)')
        axs[0].grid(True)
        axs[0].legend()
        
        axs[1].plot(times, states[:, 1], label='v (velocity)')
        axs[1].set_ylabel('v (velocity)')
        axs[1].set_xlabel('Time (s)')
        axs[1].grid(True)
        axs[1].legend()
        
        axs[2].plot(times, states[:, 2], label='theta (angle)')
        axs[2].set_ylabel('theta (angle)')
        axs[2].set_xlabel('Time (s)')
        axs[2].grid(True)
        axs[2].legend()
        
        axs[3].plot(times, states[:, 3], label='omega (angular velocity)')
        axs[3].set_ylabel('omega (angular velocity)')
        axs[3].set_xlabel('Time (s)')
        axs[3].grid(True)
        axs[3].legend()
        
        axs[4].plot(times, actions[:, 0], label='u (control input)')
        axs[4].set_ylabel('u (control input)')
        axs[4].set_xlabel('Time (s)')
        axs[4].legend()
        axs[4].grid(True)

        plt.tight_layout()
        plt.show()

    # LQR Backward and Forward Recursion
    def LQRBackwardForwardRecursion(self):
        # # Define the symbolic variables using Sympy
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

        # Calculate the Jacobian and Hessian - partial derivatives
        variables = state.col_join(action)
        # F_t = del x_t f(x_t,u_t), f_t = del u_t f(x_t,u_t)
        F_t_dir = f_x_u.jacobian(variables)
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
        cl_t = cl_t.T


        ## In this particular case, the dirivatives are constant, don't actually need Symbolic functions
        ## but we need the symbolic functions when the dirivatives are not constant.
        ## But I am using the symbolic function to learn
        # F_t = np.hstack([self.Ad, self.Bd])
        # C_t = scipy.linalg.block_diag(self.Q, self.R)
        # cl_t = np.zeros((5, 1))
        # fl_t = self.Bd

        # set the initial state and action
        current_state = self.data.desired_state

        # # Define V_t+1 and v_t+1 
        # V_t_plus_1 = np.zeros((4, 4))
        # vl_t_plus_1 = np.zeros((4, 1))

        # Define cost to go at the final state as mentioned in
        # https://stanford.edu/class/ee363/lectures/dlqr.pdf
        V_t_plus_1 = self.Q
        vl_t_plus_1 = self.Q @ self.data.desired_state + self.R @ self.start_action

        # Run the backward recursion
        print("Backward Recursion")
        self.time = self.data.T
        for t in range(self.time_steps, -1, -1):
            # Calculate Q_t,q_t
            Q_t = C_t + F_t.T @ V_t_plus_1 @ F_t
            ql_t = cl_t + F_t.T @ V_t_plus_1 @ fl_t + F_t.T @ vl_t_plus_1

            ## Calculate Q(x_t,u_t) - cost togo from now to end if we take u_t for state x_t
            ## This case we are not using the Q cost function further in the calculation not commented
            #vecstate = np.concatenate((current_state, current_action), axis=0)
            #Qofx_t_u_t = 0.5 * (vecstate).T @ Q_t @ (vecstate) + vecstate.T @ ql_t
            ## Calculate u_t using u_t and x_t which gives the minimum cost
            ## u_t => argmin_u_t (Qofx_t_u_t) = K_t @ x_t + kl_t

            # Calculate Quu and Qux, Qxu and Qxx
            Quu = Q_t[4:5, 4:5]
            Qux = Q_t[4:5, 0:4]
            Qxu = Q_t[0:4, 4:5]
            Qxx = Q_t[0:4, 0:4]
            qx = ql_t[0:4, 0]
            qu = ql_t[4:5, 0]

            # Calculate K_t and k_t. Add regularization to Quu to avoid issues with inverse
            #Quu_reg = Quu + 1 * np.eye(Quu.shape[0])
            #K_t = -np.linalg.solve(Quu_reg, Qux)
            #kl_t = -np.linalg.solve(Quu_reg, qu)
            # K_t = (-np.linalg.inv(Quu_reg) @ Qux)
            # kl_t = (-np.linalg.inv(Quu_reg) @ qu)
            
            # Calculate the inverse of Quu, Inverse is a delicate operation and can easily overflow or underflow.
            Quu_inv = np.linalg.inv(Quu)
            K_t = -Quu_inv @ Qux
            kl_t = -Quu_inv @ qu

            # Calculate V_t and vl_t. From class calculate V_t and vl_t
            V_t = Qxx + Qxu @ K_t + K_t.T @ Qux + K_t.T @ Quu @ K_t
            vl_t = qx + Qxu @ kl_t + K_t.T @ qu + K_t.T @ Quu @ kl_t

            # With V_t and vl_t dependent on K_t and kl_t (in turn have inverse call)
            # The values can cause overflow or underflow. So we need to clip the values for stabilization
            # set the clip values to a any reasonable value
            #V_t = 0.5 * (V_t + V_t.T)
            vl_t = np.clip(vl_t, -1e0, 1e0)

            # Calculate Vofx_t - cost togo from now untill end if from state x_t
            Vofx_t = 0.5 * current_state.T @ V_t @ current_state + current_state.T @ vl_t

            # set the V_t_plus_1 and vl_t_plus_1 for the next iteration
            V_t_plus_1 = V_t
            vl_t_plus_1 = vl_t
            
            # Store the results. Can be used directly but saving for easier debugging and the problem is small
            #print(f"Current State: {current_state}, Current Action: {current_action}, Next State: {next_state}")
            # output = {
            #     'time': t,
            #     'K_t': K_t,
            #     'kl_t': kl_t,
            # }

            # self.output.insert(0,output)
            self.time -= self.data.dt
        
        print("Forward Recursion")
        # Run the forward recursion
        current_state = self.data.initial_state
        self.time = 0
        for t in range(self.time_steps):
            # Pull the latest values of K_t and kl_t to calculate the action as the system is time invariant
            # K_t = self.output[0]['K_t']
            # kl_t = self.output[0]['kl_t']
            u_t = K_t @ current_state + kl_t
            action = np.array([[u_t]], dtype=np.float64).reshape(1, 1)
            next_state = self.lin_dynamics(current_state, action)
            # Store the results
            output = {
                'time': t,
                'state': current_state,
                'K_t': K_t,
                'kl_t': kl_t,
                'u_t': u_t,
            }
            current_state = next_state
            self.time += self.data.dt
            self.outputwithaction.append(output)

        return self.outputwithaction 

if __name__ == "__main__":
    data = simData()
    pendulum = Pendulum(data)
    output = pendulum.LQRBackwardForwardRecursion()

    # Simulate the pendulum dynamics
    trajectory = pendulum.simulate(output=output)
    with open("pendulum_simulation.txt", "w") as f:
        for step in trajectory:
            f.write(f"Time: {step['time']}, State: {step['state']}, Action: {step['action']}, Next State: {step['next_state']}\n")

    # Plot the results
    pendulum.plot_simulation_results(trajectory)
    #pendulum.plot_simulation_results_separately(trajectory)