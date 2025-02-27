import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import joblib
import os
from tqdm import tqdm
import time
import const
import utilities as ut
import mockSQLenv as SQLenv
import agent as agn

def train_model_on_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Training on: {device}")

    nepisodes = 100
    exploration = 0.9
    min_exploration = 0.05
    decay = 0.95

    agt = agn.Agent(const.actions, verbose=False)
    agt.set_learning_options(
        exploration=exploration, learningrate=0.1, discount=0.9, max_step=1000
    )

    # Chuyển Q-table sang Tensor GPU
    agt.Q = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in agt.Q.items()}

    steps, rewards, states = [], [], []
    for _ in tqdm(range(nepisodes)):
        env = SQLenv.mockSQLenv(
            verbose=False, data_reward=10, syntax_reward=-5,
            differ_col_reward=-1, query_reward=-2, waf_block=-20
        )
        
        agt.reset(env)
        agt.run_episode()
        
        steps.append(agt.steps)
        rewards.append(agt.rewards)
        states.append(ut.getdictshape(agt.Q)[0])
        
        # Giảm exploration theo decay
        exploration = max(min_exploration, exploration * decay)
        agt.set_learning_options(exploration=exploration)

    print("Done training")
    file_path = 'q_table_trained.pkl'
    if os.path.exists(file_path):
        os.remove(file_path)

    # Chuyển Q-table từ Tensor về NumPy để lưu trữ
    q_table_cpu = {key: value.cpu().numpy() for key, value in agt.Q.items()}
    joblib.dump(q_table_cpu, file_path)

    return agt, steps, rewards, states

if __name__ == "__main__":
    start_time = time.time()
    agent_trained, steps, rewards, states = train_model_on_gpu()
    print("Steps =", steps)
    print("Rewards =", rewards)
    print("States =", states)
    end_time = time.time()
    print(f"Thời gian thực thi: {end_time - start_time:.6f} giây")