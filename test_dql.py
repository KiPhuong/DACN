import torch
import const
import mockSQLenv as srv
from agent_dql import Agent

def test_agent(model_path="dqn_agent.pth", num_tests=1000):
    env = srv.mockSQLenv(verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(const.actions, verbose=True)
    agent.device = device
    agent.model.to(device)

    # Load lại mô hình đã huấn luyện
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()  # chuyển sang chế độ inference (tắt dropout, batchnorm nếu có)

    print(f"Trainning on {device}...")

    successes = 0
    for i in range(num_tests):
        print(f"\n----- Test case {i+1} -----")
        env = srv.mockSQLenv(verbose=True)
        agent.reset(env)
        result = agent.run_episode(env)
        if result:
            print("True")
            successes += 1
        else:
            print("False")
    
    print(f"\nTrue time: {successes}/{num_tests}")

if __name__ == "__main__":
    test_agent()
