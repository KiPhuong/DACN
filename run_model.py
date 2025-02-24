import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as SGfilter
from IPython.display import clear_output
import datetime
import joblib
from tqdm import tqdm
import requests
from bs4 import BeautifulSoup

import const
import utilities as ut
import mockSQLenv as SQLenv
import agent as agn



def train_model(): 
    nepisodes = 10000
    agt = agn.Agent(const.actions,verbose=False)
    agt.set_learning_options(exploration = 0.5, 
                         learningrate=0.1, 
                         discount=0.9, max_step = 100)

    steps = []; rewards = []; states = []
    for _ in tqdm(range(nepisodes)):
        env = SQLenv.mockSQLenv(verbose=False, data_reward = 10, syntax_reward = -5, differ_col_reward = -1, query_reward = -2, waf_block= -20)
    
        agt.reset(env)
        agt.run_episode()
    
        steps.append(agt.steps)
        rewards.append(agt.rewards)
        states.append(ut.getdictshape(agt.Q)[0]) 

    print("Done train")

    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S%f")
    joblib.dump(agt.Q,'ignore_zixem.pkl')

    return agt, steps, rewards, states

def test_model():
    nepisodes = 1000
    agt = agn.Agent(const.actions,verbose=False)
    agt.Q = joblib.load('ignore_zixem.pkl') 
    print (agt.Q)
    agt.set_learning_options(exploration=0.01, 
                         learningrate=0.1, 
                         discount=0.9, max_step = 100)

    Tsteps = []; Trewards = []; Tstates = []
    for _ in tqdm(range(nepisodes)):
        env = SQLenv.mockSQLenv(verbose=True, data_reward = 10, syntax_reward = -5, differ_col_reward = -1, query_reward = -2, waf_block = -20)
    
        agt.reset(env)
        agt.run_episode()
    
        Tsteps.append(agt.steps)
        Trewards.append(agt.rewards)
        Tstates.append(ut.getdictshape(agt.Q)[0]) 

if __name__ == "__main__":
    agent_trained, steps, rewards, states = train_model()
    print("Steps = ", steps)
    print("Rewards = ", rewards)
    print("States = ", states)
    test_model()


