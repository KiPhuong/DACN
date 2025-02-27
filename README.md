# CTF-SQL
Modelling SQL Injection Using Reinforcement Learning

### Requirements
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import savgol_filter as SGfilter
from IPython.display import clear_output
import datetime
import joblib
from tqdm import tqdm
import time
import requests
from bs4 import BeautifulSoup
import os
import const
import utilities as ut
import mockSQLenv as SQLenv
import agent as agn
import torch
import torch.nn as nn
import torch.optim as optim



### Content
Using RL to exploit web zixem lv1 without WAF (custom)
q_table_trained.pkl == luu bang q-table sau khi train
train_model.py == train model tren cpu voi web khong cau hinh waf
test_model.py == test model tren cpu voi web khong cau hinh waf
train_model_on_gpu.py == train model tren gpu voi web khong cau hinh waf (chua test)
test_model_on_gpu.py == test model tren gpu voi web khong cau hinh waf (chua test)\
agent.py == cau hinh agent cho RL
mockSQLenv.py == phan hoi cua moi truong khi agent tuong tac voi web
generation_actions.py == tao action cho agent 

### References

\[1\] Erdodi, L., Sommervoll, A.A. and Zennaro, F.M., 2020. Simulating SQL Injection Vulnerability Exploitation Using Q-Learning Reinforcement Learning Agents. arXiv preprint.

https://github.com/avalds/gym-CTF-SQL
