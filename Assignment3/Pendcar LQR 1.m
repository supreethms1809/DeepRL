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
Q = eye(4);
R = 0.01;


% Convert continuous-time dynamics to discrete-time dynamics using c2d function
% Ad and Bd are the matrices of the resulting discrete-time dynamics
[Ad,Bd]=c2d(A,B,dt);

F_t = [Ad, Bd;];
fl_t = Bd;
C_t = blkdiag(Q, R);
cl_t = zeros(5,1); % Cost function linear term

% K = cell(T/dt, 1);
% kl = cell(T/dt, 1);

V_t_plus_1 = zeros(4,4);
vl_t_plus_1 = zeros(4,1);

% LQR backward recursion
for t = T/dt:-1:1
  
    %%%%%%% Complete backward recursion here %%%%%%%%%%
    Q_t = C_t + F_t' * V_t_plus_1 * F_t;
    ql_t = cl_t + F_t' * V_t_plus_1 * fl_t + F_t' * vl_t_plus_1;

    Quu = Q_t(5:5, 5:5);    
    Qux = Q_t(5:5, 1:4);
    Qxu = Q_t(1:4, 5:5);
    Qxx = Q_t(1:4, 1:4);
    qx = ql_t(1:4);
    qu = ql_t(5:5);

    Quu_inv = inv(Quu);
    K_t = -Quu_inv * Qux;
    kl_t = -Quu_inv * qu;

    % % Store the feedback gain and feedforward term
    % K{t} = K_t;
    % kl{t} = kl_t;

    V_t = Qxx + Qxu * K_t + K_t' * Qux + K_t' * Quu * K_t;
    vl_t = qx + Qxu * kl_t + K_t' * qu + K_t' * Quu * kl_t;

    V_t_plus_1 = V_t;
    vl_t_plus_1 = vl_t;

end

state = x0; % Initial state
% LQR forward recursion
for t = 1:T/dt

    %%%%%%% Complete forward recursion here %%%%%%%%%%
    % K_t = K{t}; % Feedback gain for current time step
    % kl_t = kl{t}; % Feedforward term for current time step
    u_t = K_t * (state) + kl_t; % Control input
    state = Ad * state + Bd * u_t; % Update state using discrete-time dynamics
    % Store state for plotting
    states(:, t) = state;
    actions(:, t) = u_t;

end

actions_clipped = max(min(actions, 200), -200);

% Plot results
figure;

% Time vector
time = 0:dt:T-dt;

% Plot all states and control input on the same graph
plot(time, states(1, :), 'r', 'LineWidth', 2, 'DisplayName', 'Position (x)');
hold on;
plot(time, states(2, :), 'g', 'LineWidth', 2, 'DisplayName', 'Velocity (v)');
plot(time, states(3, :), 'b', 'LineWidth', 2, 'DisplayName', 'Angle (theta)');
plot(time, states(4, :), 'k', 'LineWidth', 2, 'DisplayName', 'Angular Velocity (omega)');
plot(time, actions_clipped(1,:), 'm', 'LineWidth', 2, 'DisplayName', 'Control Input (u)');

% Add labels, title, legend, and grid
xlabel('Time (s)');
ylabel('Values');
title('Pendulum Simulation Results');
legend('show'); % Automatically show all labeled plots in the legend
grid on;
hold off;
