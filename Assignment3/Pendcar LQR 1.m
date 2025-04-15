clear all, close all, clc
% System dynamics 
m = 1; M = 5; L = 2; g = -10; d = 1;
b = 1; % Pendulum up fixed point(b=1)
A = [0 1 0 0;
0 -d/M b*m*g/M 0;
0 0 0 1;
0 -b*d/(M*L) -b*(m+M)*g/(M*L) 0];
B = [0; 1/M; 0; b*1/(M*L)];


% Parameters 
T = 20; % Simulation time (s)
dt = 0.001; % Sampling period (s)
wr = [0,0,0,0]'; % Desired state
x0 = [2,1,2,1]'; % initial state


% Convert continuous-time dynamics to discrete-time dynamics using c2d function
% Ad and Bd are the matrices of the resulting discrete-time dynamics
[Ad,Bd]=c2d(A,B,dt);



% LQR backward recursion
for t = T/dt:-1:1
  
    %%%%%%% Complete backward recursion here %%%%%%%%%%


end


% LQR forward recursion
for t = 1:T/dt

    %%%%%%% Complete forward recursion here %%%%%%%%%%


end



% Plot results
