import torch
import os
import time
import numpy as np
import csv
from collections import Counter

from agent_dql import Agent
from env import mockSQLenv as srv
from env import const
from env import utilities as ut

def train_agent(num_episodes, num_measure, save_model=True, model_path="dqn_agent.pth", verbose=True, csv_path="metrics.csv"):
    env = srv.mockSQLenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    agent = Agent(const.actions, verbose=False)
    agent.model.to(device)
    
    steps, rewards, successes, trajectories = [], [], [], []
    failure_reasons = {"max_step": 0, "waf_block": 0, "syntax_error": 0, "other": 0}
    episode_times = []
    total_rewards = 0
    total_ep_action = []  # Lưu tất cả hành động qua các episode
    start_time = time.time()

    # CSV headers
    csv_headers = [
        "Episode", "Success_Rate", "Average_Queries", "Average_Reward", "Average_Time",
        "Average_Loss", "Failures_Max_Step", "Failures_WAF_Block", "Failures_Syntax_Error",
        "Failures_Other", "Action_Distribution", "Cumulative_Action_Distribution"
    ]
    if os.path.exists(csv_path):
        os.remove(csv_path)
    with open(csv_path, mode='w', newline='') as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(csv_headers)
  
    # Initialize log files
    time_path_output = "time_output.txt"
    weights_log_path = "model_weights_log.txt"
    loss_log_path = "model_loss_log.txt"

    if os.path.exists(time_path_output):
        os.remove(time_path_output)
    if os.path.exists(weights_log_path):
        os.remove(weights_log_path)
    if os.path.exists(loss_log_path):
        os.remove(loss_log_path)

    for episode in range(num_episodes):
        episode_start = time.time()
        env = srv.mockSQLenv(
            verbose=False, data_reward=20, syntax_reward=-10,
            differ_col_reward=-10, query_reward=-10, waf_block=-30
        )
        agent.reset(env)
        agent.device = device
        
        terminated, episode_steps, episode_rewards, _, episode_action_used = agent.run_episode(env)
        
        steps.append(episode_steps)
        rewards.append(episode_rewards)
        total_rewards += episode_rewards
        total_ep_action += episode_action_used
        
        if terminated and episode_steps < agent.max_step:
            successes.append(1)
        else:
            successes.append(0)
            last_response = env.debug_msg if hasattr(env, 'debug_msg') else ""
            if episode_steps >= agent.max_step:
                failure_reasons["max_step"] += 1
            elif "WAF block" in last_response:
                failure_reasons["waf_block"] += 1
            elif "error in your SQL syntax" in last_response:
                failure_reasons["syntax_error"] += 1
            else:
                failure_reasons["other"] += 1

        if (episode + 1) % num_measure == 0:
            trajectories.append(total_ep_action)

        episode_end = time.time()
        episode_time = episode_end - episode_start  
        episode_times.append(episode_time)

        with open(time_path_output, mode='a') as time_file:
            time_file.write(f"{episode_time:.4f}\n")

        if (episode + 1) % num_measure == 0:
            success_rate = np.mean(successes[-num_measure:]) * 100
            avg_queries = np.mean(steps[-num_measure:])
            avg_reward = np.mean(rewards[-num_measure:])
            avg_time = np.mean(episode_times[-num_measure:])
            avg_loss = np.mean(agent.loss_history[-num_measure:]) if agent.loss_history else 0.0

            # Action distribution cho episode hiện tại
            action_dist = ""
            if trajectories:
                last_trajectory = trajectories[-1]
                action_counts = {act: last_trajectory.count(act) for act in set(last_trajectory)}
                action_dist = str(action_counts)

            # Cumulative action distribution qua tất cả episode
            cumulative_action_counts = dict(Counter(total_ep_action))
            cumulative_action_dist = str(cumulative_action_counts)

            print(f"\nEpisode {episode + 1}/{num_episodes}")
            print(f"Success Rate (last {num_measure}): {success_rate:.2f}%")
            print(f"Average Queries (last {num_measure}): {avg_queries:.2f}")
            print(f"Average Reward (last {num_measure}): {avg_reward:.2f}")
            print(f"Average Time (last {num_measure}): {avg_time:.4f} seconds")
            print(f"Average Loss (last {num_measure}): {avg_loss:.6f}")
            print(f"Failure Analysis: {failure_reasons}")
            print(f"Trajectory Behavior (Action Distribution): {action_dist}")
            print(f"Cumulative Action Distribution: {cumulative_action_dist}")

            with open(weights_log_path, mode='a') as weight_file:
                weight_file.write(f"\n--- Episode {episode + 1} ---\n")
                for name, param in agent.model.named_parameters():
                    if param.grad is not None:
                        mean_weight = param.data.mean().item()
                        std_weight = param.data.std().item()
                        mean_grad = param.grad.mean().item()
                        std_grad = param.grad.std().item()
                        weight_file.write(f"Layer {name}:\n")
                        weight_file.write(f"  Weight mean: {mean_weight:.6f}, std: {std_weight:.6f}\n")
                        weight_file.write(f"  Gradient mean: {mean_grad:.6f}, std: {std_grad:.6f}\n")
                    else:
                        weight_file.write(f"Layer {name}: No gradient available\n")

            with open(loss_log_path, mode='a') as loss_file:
                loss_file.write(f"Episode {episode + 1}: Average Loss = {avg_loss:.6f}\n")

            with open(csv_path, mode='a', newline='') as csv_file:
                writer = csv.writer(csv_file)
                writer.writerow([
                    episode + 1,
                    f"{success_rate:.2f}",
                    f"{avg_queries:.2f}",
                    f"{avg_reward:.2f}",
                    f"{avg_time:.4f}",
                    f"{avg_loss:.6f}",
                    failure_reasons["max_step"],
                    failure_reasons["waf_block"],
                    failure_reasons["syntax_error"],
                    failure_reasons["other"],
                    action_dist,
                    cumulative_action_dist
                ])

            if verbose:
                print(f"Steps (last {num_measure}): {steps[-num_measure:][:10]}...")
                print(f"Rewards (last {num_measure}): {rewards[-num_measure:][:10]}...")
                print(f"Loss (last {num_measure}): {agent.loss_history[-num_measure:][:10]}...")
                for name, param in agent.model.state_dict().items():
                    print(name, param.shape)
                for name, param in agent.model.named_parameters():
                    if param.requires_grad:
                        print(f"{name} - First 5 values: {param.data.view(-1)[:5]}")

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
    print(f"Model weights and gradients saved to {weights_log_path}")
    print(f"Loss history saved to {loss_log_path}")

if __name__ == "__main__":
    train_agent(num_episodes=10000, num_measure=1000, verbose=False)