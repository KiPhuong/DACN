import numpy as np
import torch
import joblib
from tqdm import tqdm
import time
import const
import utilities as ut
import mockSQLenv as SQLenv
import agent_on_gpu as agn

def test_model_on_gpu():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Testing on: {device}")

    nepisodes = 20000
    agt = agn.Agent(const.actions, verbose=False)
    
    # Load Q-table and move to GPU
    q_table_cpu = joblib.load('q_table_trained_on_gpu.pkl')
    agt.Q = {key: torch.tensor(value, dtype=torch.float32, device=device) for key, value in q_table_cpu.items()}
    
    agt.set_learning_options(exploration=0.01, learningrate=0.1, discount=0.9, max_step=100)
    
    Tsteps, Trewards, Tstates = [], [], []
    for _ in tqdm(range(nepisodes)):
        env = SQLenv.mockSQLenv(verbose=False, data_reward=10, syntax_reward=-5, differ_col_reward=-1, query_reward=-2, waf_block=-20)
    
        agt.reset(env)
        agt.run_episode()
        
        print("Episode test steps =", agt.steps)
        print("Episode test rewards =", agt.rewards)
        
        Tsteps.append(agt.steps)
        Trewards.append(agt.rewards)
        Tstates.append(ut.getdictshape(agt.Q)[0])

if __name__ == "__main__":
    start_time = time.time()
    test_model_on_gpu()
    end_time = time.time()
    print(f"Thời gian thực thi: {end_time - start_time:.6f} giây")

