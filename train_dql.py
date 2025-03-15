import torch
import const
import mockSQLenv as srv
from agent_dql import Agent

def train_agent(num_episodes=1000, save_model=True, model_path="dqn_agent.pth"):
    
    env = srv.mockSQLenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    agent = Agent(const.actions, verbose=False)
    agent.model.to(device)
    steps, rewards, states = [], [], []
    for episode in range(num_episodes):
        env = srv.mockSQLenv(
            verbose=False, data_reward=10, syntax_reward=-5,
            differ_col_reward=-1, query_reward=-2, waf_block=-20
        )
        agent.reset(env)
        agent.device = device
        agent.run_episode(env)
        steps.append(agent.steps)
        rewards.append(agent.rewards)

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: Total rewards = {agent.rewards}")
            print(steps)
            print(rewards)
            print(agent.model)
            for name, param in agent.model.state_dict().items():
                print(name, param.shape)
    if save_model:
        torch.save(agent.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    

if __name__ == "__main__":
    train_agent(num_episodes=20000)
