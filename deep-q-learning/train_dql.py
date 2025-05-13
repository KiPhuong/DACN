import torch
import os
import time
import numpy as np
import csv

from agent_dql import Agent
from env import mockSQLenv as srv
from env import const

def train_agent(num_episodes, save_model=True, model_path="dqn_agent.pth", verbose=True, csv_path="metrics.csv"):
    env = srv.mockSQLenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    agent = Agent(const.actions, verbose=False)
    agent.model.to(device)
    
    # Lists to store metrics
    steps, rewards, successes, trajectories = [], [], [], []
    failure_reasons = {"max_step": 0, "waf_block": 0, "syntax_error": 0, "other": 0}
    episode_times = []
    total_rewards = 0
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

    time_path_output = "time_output.txt"
    for episode in range(num_episodes):
        episode_start = time.time()
        env = srv.mockSQLenv(
            verbose=False, data_reward=20, syntax_reward=-10,
            differ_col_reward=-7, query_reward=-7, waf_block=-30
        )
        agent.reset(env)
        agent.device = device
        
        # Track actions for trajectory analysis
        episode_actions = []
        terminated, episode_steps, episode_rewards = agent.run_episode(env)
        
        # Record metrics
        steps.append(episode_steps)
        rewards.append(episode_rewards)
        total_rewards += episode_rewards
        
        # Success rate: Only count as success if terminated and not due to max_step
        if terminated and episode_steps < agent.max_step:
            successes.append(1)
        else:
            successes.append(0)
            # Failure analysis
            last_response = env.debug_msg if hasattr(env, 'debug_msg') else ""
            if episode_steps >= agent.max_step:
                failure_reasons["max_step"] += 1
            elif "WAF block" in last_response:
                failure_reasons["waf_block"] += 1
            elif "error in your SQL syntax" in last_response:
                failure_reasons["syntax_error"] += 1
            else:
                failure_reasons["other"] += 1

        # Trajectory: Store actions for every 1000th episode
        if (episode + 1) % 1000 == 0:
            trajectories.append(episode_actions)

        # Computational cost
        episode_end = time.time()
        episode_time = episode_end - episode_start  
        episode_times.append(episode_time)

        
        # Save episode time to time_output.txt 
        with open(time_path_output, mode='a') as time_file:
            time_file.write(f"{episode_time:.4f}\n")

        # Print and save evaluation metrics every 10000 steps
        if (episode + 1) % 10000 == 0:
            success_rate = np.mean(successes[-10000:]) * 100
            avg_queries = np.mean(steps[-10000:])
            avg_reward = np.mean(rewards[-10000:])
            avg_time = np.mean(episode_times[-10000:])

            # Action distribution for last trajectory
            action_dist = ""
            if trajectories:
                last_trajectory = trajectories[-1]
                action_counts = {act: last_trajectory.count(act) for act in set(last_trajectory)}
                action_dist = str(action_counts)

            print(f"\nEvaluation at Episode {episode + 1}/{num_episodes}:")
            print(f"Success Rate: {success_rate:.2f}%")
            print(f"Average Queries per Episode: {avg_queries:.2f}")
            print(f"Average Reward per Episode: {avg_reward:.2f}")
            print(f"Average Time per Episode: {avg_time:.4f} seconds")
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

            if verbose:
                print(f"Steps (last 10000): {steps[-10000:][:10]}...")  # Print first 10 for brevity
                print(f"Rewards (last 10000): {rewards[-10000:][:10]}...")
                for name, param in agent.model.state_dict().items():
                    print(name, param.shape)
                for name, param in agent.model.named_parameters():
                    if param.requires_grad:
                        print(f"{name} - First 5 values: {param.data.view(-1)[:5]}")

            # Reset failure reasons for next interval
            failure_reasons = {"max_step": 0, "waf_block": 0, "syntax_error": 0, "other": 0}

    if save_model:
        if os.path.exists(model_path):
            os.remove(model_path)
        torch.save(agent.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    end_time = time.time()
    print(f"Total Execution Time: {end_time - start_time:.2f} seconds")
    print(f"Overall Success Rate: {np.mean(successes) * 100:.2f}%")
    print(f"Overall Average Queries: {np.mean(steps):.2f}")
    print(f"Metrics saved to {csv_path}")
    print(f"Episode times saved to {time_path_output}")
if __name__ == "__main__":
    train_agent(num_episodes=10000, verbose=False)