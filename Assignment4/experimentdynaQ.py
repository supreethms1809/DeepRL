import os
import subprocess
import json
import matplotlib.pyplot as plt
import numpy as np
import argparse
import datetime

def plot_results_steps(steps_list, p_steps):
    episodes = np.arange(len(steps_list[0]))
    plt.figure(figsize=(10, 6))
    for i, steps in enumerate(steps_list):
        degree = 4
        coefficients = np.polyfit(episodes, steps, degree)
        trend_line = np.polyval(coefficients, episodes)
        plt.plot(episodes, steps, alpha=0.5, label=f'Raw Steps (Planning Steps={p_steps[i]})')
        if i == 0:
            plt.plot(episodes, trend_line, linestyle='-', label=f'Trend Line (Planning Steps={p_steps[i]})')
        else:
            plt.plot(episodes, trend_line, linestyle='--', label=f'Trend Line (Planning Steps={p_steps[i]})')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Steps to Goal per episode with Trend Lines for Different Planning Steps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'steps_to_goal_{dt}.png')
    plt.show()

def plot_results_rewards(steps_list, p_steps):
    episodes = np.arange(len(steps_list[0]))
    plt.figure(figsize=(10, 6))
    for i, steps in enumerate(steps_list):
        degree = 4
        coefficients = np.polyfit(episodes, steps, degree)
        trend_line = np.polyval(coefficients, episodes)
        plt.plot(episodes, steps, alpha=0.5, label=f'Raw Steps (Planning Steps={p_steps[i]})')
        if i == 0:
            plt.plot(episodes, trend_line, linestyle='-', label=f'Trend Line (Planning Steps={p_steps[i]})')
        else:
            plt.plot(episodes, trend_line, linestyle='--', label=f'Trend Line (Planning Steps={p_steps[i]})')
    plt.xlabel('Episode')
    plt.ylabel('Steps')
    plt.title('Total reward per episode with Trend Lines for Different Planning Steps')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f'rewards_to_goal_{dt}.png')
    plt.show()

def plot_grid(Q_values_list, p_steps):
    fig, axes = plt.subplots(5, 2, figsize=(10, 10))
    axes = axes.flatten()
    N = 20
    states = [(i, j) for i in range(N) for j in range(N)]
    for i, Q in enumerate(Q_values_list):
        value_grid = np.zeros((N, N))
        for state in states:
            s = f"{state}"
            value_grid[state] = max(Q[s].values())
        ax = axes[i]
        im = ax.imshow(value_grid, cmap='jet', origin='lower')
        ax.set_title(f'Q-Value Grid with planning steps: {p_steps[i]}', fontsize=8)
        ax.set_xlabel('X')
        ax.set_ylabel('Y')
        ax.set_box_aspect(1)
    for j in range(len(Q_values_list), len(axes)):
        fig.delaxes(axes[j])
    #fig.colorbar(im, ax=axes, orientation='vertical', fraction=0.02, pad=0.04)
    plt.subplots_adjust(wspace=0.4, hspace=0.6)
    plt.tight_layout()
    dt = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    plt.savefig(f"qlearning_value_tables_{dt}.png")
    plt.show()

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run dynaQ.py with different planning steps.")
    parser.add_argument("--run", action="store_true", help="Run dynaQ.py with different planning steps.")
    parser.add_argument("--plot", action="store_true", help="Plot the results.")
    parser.add_argument("--results", type=str, default="results.json", help="Path to the results file.")
    args = parser.parse_args()

if args.run:
    # Check if the results.json file exists and remove it
    # to avoid appending to an existing file, fresh run
    if os.path.exists("results.json"):
        os.remove("results.json")

    # Define the values for --planning_steps
    planning_steps_values = [0, 1, 2, 4, 8, 12, 16, 20, 24, 32]

    # Iterate over the values and run dynaQ.py for each
    for planning_steps in planning_steps_values:
        print(f"Running dynaQ.py with --planning_steps={planning_steps}")
        subprocess.run([
            "python", "dynaQ.py",
            "--grid_size", "20",
            "--alpha", "0.01",
            "--gamma", "0.99",
            "--epsilon", "0.01",
            "--episodes", "15000",
            "--loggin",
            "--steps", "1000",
            "--planning_steps", str(planning_steps)
            #"--plot"
        ])

if args.plot:
    results_file = "results.json"
    if os.path.exists(results_file):
        with open(results_file, "r") as f:
            results = json.load(f)

    steps_list = []
    p_steps = []
    rewards_list = []
    Q_values_list = []

    for l in results:
        steps_list.append(l["step_history"])
        rewards_list.append(l["reward_history"])
        p_steps.append(l["planning_steps"])
        Q_values_list.append(l["Q_table"])

    # Plot the results
    plot_results_steps(steps_list, p_steps)
    plot_results_rewards(rewards_list, p_steps)
    plot_grid(Q_values_list, p_steps)