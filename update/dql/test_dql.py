
import torch
import numpy as np
from agent_dql_updated import Agent
from env import mockSQLenv as srv
from env import const

def test_agent(model_path="dqn_agent.pth", num_episodes=100, verbose=True):
    print("Loading model from", model_path)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(const.actions, verbose=verbose, exploration=0.0)  # no exploration during test
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.to(device)

    successes = 0
    steps_list = []
    rewards_list = []

    for episode in range(num_episodes):
        env = srv.mockSQLenv(verbose=False)
        agent.reset(env)
        terminated, steps, reward, actions = agent.run_episode(env)

        steps_list.append(steps)
        rewards_list.append(reward)
        if terminated:
            successes += 1

        if verbose:
            print(f"Episode {episode+1}: {'SUCCESS' if terminated else 'FAILURE'} | Steps: {steps} | Reward: {reward}")
            print("Actions:", actions)
            print("-" * 50)

    success_rate = 100 * successes / num_episodes
    avg_steps = np.mean(steps_list)
    avg_reward = np.mean(rewards_list)

    print("\n===== TEST SUMMARY =====")
    print(f"Model: {model_path}")
    print(f"Test Episodes: {num_episodes}")
    print(f"Success Rate: {success_rate:.2f}%")
    print(f"Average Steps: {avg_steps:.2f}")
    print(f"Average Reward: {avg_reward:.2f}")

if __name__ == "__main__":
    test_agent(model_path="dqn_agent.pth", num_episodes=100, verbose=True)
