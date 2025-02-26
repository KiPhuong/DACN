import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as SGfilter
from IPython.display import clear_output
import datetime
import joblib
from tqdm import tqdm
#import requests
#from bs4 import BeautifulSoup
import os
import const
import utilities as ut
import mockSQLenv as SQLenv
import agent as agn



def train_model(): 
    nepisodes = 100000

    exploration = 0.9
    min_exploration = 0.05
    decay = 0.95
    agt = agn.Agent(const.actions,verbose=False)
    agt.set_learning_options(exploration = exploration, 
                         learningrate=0.1, 
                         discount=0.9, max_step = 1000)

    steps = []; rewards = []; states = []
    for _ in tqdm(range(nepisodes)):
        env = SQLenv.mockSQLenv(verbose=False, data_reward = 10, syntax_reward = -5, differ_col_reward = -1, query_reward = -2, waf_block= -20)
    
        agt.reset(env)
        agt.run_episode()
    
        steps.append(agt.steps)
        rewards.append(agt.rewards)
        states.append(ut.getdictshape(agt.Q)[0]) 
        #giam dan gia tri exploration
        exploration = max(min_exploration, exploration * decay)
        agt.set_learning_options(exploration=exploration)

    print("Done train")
    file_path = 'q_table_trained.pkl' 
    # Xóa file cũ nếu tồn tại
    if os.path.exists(file_path):
        os.remove(file_path)

    #Save q-table
    joblib.dump(agt.Q,'q_table_trained.pkl')

    return agt, steps, rewards, states

def test_model():
    nepisodes = 10000
    agt = agn.Agent(const.actions,verbose=False)
    agt.Q = joblib.load('q_table_trained.pkl') 
    #print (agt.Q)
    agt.set_learning_options(exploration=0.01, 
                         learningrate=0.1, 
                         discount=0.9, max_step = 100)

    Tsteps = []; Trewards = []; Tstates = []
    for _ in tqdm(range(nepisodes)):
        env = SQLenv.mockSQLenv(verbose=False, data_reward = 10, syntax_reward = -5, differ_col_reward = -1, query_reward = -2, waf_block = -20)
    
        agt.reset(env)
        agt.run_episode()
        print("Test steps = ", agt.steps)
        Tsteps.append(agt.steps)
        Trewards.append(agt.rewards)
        Tstates.append(ut.getdictshape(agt.Q)[0]) 

if __name__ == "__main__":
    agent_trained, steps, rewards, states = train_model()
    print("Steps = ", steps)
    print("Rewards = ", rewards)
    print("States = ", states)
    test_model()


