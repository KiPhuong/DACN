import torch
import const
import mockSQLenv as srv
from agent_dql import Agent
import time
import sys
from contextlib import redirect_stdout

def test_agent(model_path="dqn_agent.pth", num_tests=10000):
    env = srv.mockSQLenv(verbose=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agent = Agent(const.actions, exploration=0.05, verbose=False)
    agent.device = device
    agent.model.to(device)

    # Load lại mô hình đã huấn luyện
    agent.model.load_state_dict(torch.load(model_path, map_location=device))
    agent.model.eval()  # chuyển sang chế độ inference (tắt dropout, batchnorm nếu có)

    with open("output_test_dql.txt", "w", encoding="utf-8") as file:
        with redirect_stdout(file):
            print(f"Testing on {device}...")

            for i in range(num_tests):
                start_time = time.time()
                print(f"\n----- Test case {i+1} -----")
                env = srv.mockSQLenv(verbose=False, data_reward=20, syntax_reward=-10, differ_col_reward=-10, query_reward=-10, waf_block=-30)
                agent.reset(env)
                terminated, steps, rewards = agent.run_episode(env)

                if terminated:
                    print(f"Steps: {steps} | Rewards: {rewards}")
                else:
                    print(f"False test case {i + 1}")

                elapsed = time.time() - start_time
                print(f"{i+1}/{num_tests} [{'='*30}] - {elapsed/steps:.1f}s/step - reward-avg: {rewards/steps:.1f}")

if __name__ == "__main__":
    test_agent()
