import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as SGfilter
from IPython.display import clear_output
import datetime
import joblib
from tqdm import tqdm
import time
#import requests
#from bs4 import BeautifulSoup
import os
import const
import utilities as ut
import mockSQLenv as SQLenv
import agent as agn


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
        env = SQLenv.mockSQLenv(verbose=True, data_reward = 10, syntax_reward = -5, differ_col_reward = -1, query_reward = -2, waf_block = -20)
    
        agt.reset(env)
        agt.run_episode()
        print("Episode test steps = ", agt.steps)
        print("Episode test rewards = ", agt.rewards)
        Tsteps.append(agt.steps)
        Trewards.append(agt.rewards)
        Tstates.append(ut.getdictshape(agt.Q)[0]) 

if __name__ == "__main__":
    start_time = time.time()
    test_model()
    end_time = time.time()
    print(f"Thời gian thực thi: {end_time - start_time:.6f} giây")