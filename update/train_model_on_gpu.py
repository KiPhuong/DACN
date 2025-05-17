import numpy as np
import torch
import os
import time
import csv
from tqdm import tqdm

import agent_on_gpu as agn
from env import mockSQLenv as SQLenv
from env import const
from env import utilities as ut

def train_model_on_gpu(num_episodes=10000, num_measure=1000, save_model=True, model_path="q_table_trained_on_gpu.pkl", verbose=False, csv_path="metrics_gpu.csv"):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    exploration = 1.5
    min_exploration = 0.3
    decay = 0.99995

    agt = agn.Agent(const.actions, verbose=False)
    agt.set_learning_options(
        exploration=exploration, learningrate=0.1, discount=0.9, max_step=1000
    )

    # Convert Q-table to GPU Tensor
    agt.Q = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in agt.Q.items()}

    # Lists to store metrics
    steps, rewards, successes, trajectories = [], [], [], []
    failure_reasons = {"max_step": 0, "waf_block": 0, "syntax_error": 0, "other": 0}
    episode_times = []
    total_rewards = 0
    total_ep_action = []
    start_time = time.time()

    # Initialize CSV file
    csv_headers = [
        "Episode", "Success_Rate", "Average_Queries", "Average_Reward", "Average_Time",
        "Failures_Max_Step", "Failures_WAF_Block", "Failures_Syntax_Error", "Failures_Other",
        "Action_Distribution"
    ]
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)

    time_path_output = "time_output_qlearning.txt"
    if os.path.exists(time_path_output):
        os.remove(time_path_output)

    for episode in tqdm(range(num_episodes)):
        episode_start = time.time()
        env = SQLenv.mockSQLenv(
            verbose=False, data_reward=10, syntax_reward=-10,
            differ_col_reward=-10, query_reward=-10, waf_block=-20
        )
        
        agt.reset(env)
        # Track actions for trajectory analysis
        episode_actions = []
        terminated, episode_steps, episode_rewards, episode_actions = agt.run_episode()
        
        # Record metrics
        steps.append(episode_steps)
        rewards.append(episode_rewards)
        total_rewards += episode_rewards

        # Giảm exploration theo decay
        exploration = max(min_exploration, exploration * decay)
        agt.set_learning_options(exploration=exploration)

        # Success rate: Only count as success if terminated and not due to max_step
        if terminated and episode_steps < agt.max_step:
            successes.append(1)
        else:
            successes.append(0)
            # Failure analysis
            last_response = env.debug_msg if hasattr(env, 'debug_msg') else ""
            if episode_steps >= agt.max_step:
                failure_reasons["max_step"] += 1
            elif "WAF block" in last_response:
                failure_reasons["waf_block"] += 1
            elif "error in your SQL syntax" in last_response:
                failure_reasons["syntax_error"] += 1
            else:
                failure_reasons["other"] += 1



        # Computational cost
        episode_end = time.time()
        episode_time = episode_end - episode_start
        episode_times.append(episode_time)

        # Save episode time to time_output_gpu.txt
        with open(time_path_output, mode='a') as time_file:
            time_file.write(f"{episode_time:.4f}\n")

        # Trajectory: Store actions for every num_measure episode
        if (episode + 1) % num_measure == 0:
            trajectories.append(episode_actions)

        # Print and save evaluation metrics every num_measure steps
        if (episode + 1) % num_measure == 0:
            success_rate = np.mean(successes[-num_measure:]) * 100
            avg_queries = np.mean(steps[-num_measure:])
            avg_reward = np.mean(rewards[-num_measure:])
            avg_time = np.mean(episode_times[-num_measure:])

            # Action distribution for last trajectory
            action_dist = ""
            if trajectories:
                last_trajectory = trajectories[-1]
                action_counts = {act: last_trajectory.count(act) for act in set(last_trajectory)}
                action_dist = str(action_counts)

            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Success Rate (last {num_measure}): {success_rate:.2f}%")
            print(f"Average Queries (last {num_measure}): {avg_queries:.2f}")
            print(f"Average Reward (last {num_measure}): {avg_reward:.2f}")
            print(f"Average Time (last {num_measure}): {avg_time:.4f} seconds")
            print(f"Failure Analysis: {failure_reasons}")
            print(f"Trajectory Behavior (Action Distribution): {action_dist}")

            # Save metrics to CSV
            with open(csv_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    episode + 1,
                    f"{success_rate:.2f}",
                    f"{avg_queries:.2f}",
                    f"{avg_reward:.2f}",
                    f"{avg_time:.4f}",
                    failure_reasons["max_step"],
                    failure_reasons["waf_block"],
                    failure_reasons["syntax_error"],
                    failure_reasons["other"],
                    action_dist
                ])

            if True:
                print(f"Steps (last {num_measure}): {steps[-num_measure:][:10]}...")
                print(f"Rewards (last {num_measure}): {rewards[-num_measure:][:10]}...")
                # Print Q-table stats
                for key, value in list(agt.Q.items())[:5]:  # Print first 5 for brevity
                    print(f"Q[{key}] shape: {value.shape}, First 5 values: {value.flatten()[:5]}")

            # Reset failure reasons for next interval
            failure_reasons = {"max_step": 0, "waf_block": 0, "syntax_error": 0, "other": 0}

    if save_model:
        if os.path.exists(model_path):
            os.remove(model_path)
        # Convert Q-table from Tensor to NumPy for storage
        q_table_cpu = {
            key: (value.cpu().numpy() if isinstance(value, torch.Tensor) else value)
            for key, value in agt.Q.items()
        }
        joblib.dump(q_table_cpu, model_path)
        print(f"Model saved to {model_path}")

    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Overall Success Rate: {np.mean(successes) * 100:.2f}%")
    print(f"Overall Average Queries: {np.mean(steps):.2f}")
    print(f"Metrics saved to {csv_path}")
    print(f"Episode times saved to {time_path_output}")

    return agt, steps, rewards, successes

if __name__ == "__main__":
    start_time = time.time()
    agent_trained, steps, rewards, successes = train_model_on_gpu(verbose=False)
    print("Steps =", steps)
    print("Rewards =", rewards)
    print("Successes =", successes)
    print("Q table = ", {k: v.shape for k, v in agent_trained.Q.items()})
    end_time = time.time()
    print(f"Thời gian thực thi: {end_time - start_time:.6f} giây")