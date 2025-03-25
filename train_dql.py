import torch
import const
import os
import mockSQLenv as srv
from agent_dql import Agent

def train_agent(num_episodes, save_model=True, model_path="dqn_agent.pth"):
    
    env = srv.mockSQLenv()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    agent = Agent(const.actions, verbose=False)
    agent.model.to(device)
    steps, rewards = [], []
    total_rewards = 0
    for episode in range(num_episodes):
        
        env = srv.mockSQLenv(
            verbose=False, data_reward=20, syntax_reward=-10,
            differ_col_reward=-7, query_reward=-7, waf_block=-30
        )
        agent.reset(env)
        agent.device = device
        agent.run_episode(env)
        steps.append(agent.steps)
        rewards.append(agent.rewards)
        total_rewards += agent.rewards

        if (episode + 1) % 100 == 0:
            print(f"Episode {episode + 1}/{num_episodes}: Total rewards = {total_rewards}")
            
            print(steps)
            print(rewards)
            #print(agent.model)
            steps, rewards = [], []
            total_rewards = 0
            for name, param in agent.model.state_dict().items():
                print(name, param.shape)
            for name, param in agent.model.named_parameters():
                if param.requires_grad:
                    print(f"{name} - giá trị đầu tiên: {param.data.view(-1)[:5]}")
    if save_model:
        if os.path.exists(model_path):
            os.remove(model_path)

        torch.save(agent.model.state_dict(), model_path)
        print(f"Model saved to {model_path}")

    

if __name__ == "__main__":
    train_agent(num_episodes=1000)
