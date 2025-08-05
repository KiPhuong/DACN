import numpy as np
import sys
import torch
import requests
from bs4 import BeautifulSoup

import mockSQLenv as srv
import const
import utilities as ut

"""
agent.py is based on FMZennaro's agent on https://github.com/FMZennaro/CTF-RL/blob/master/Simulation1/agent.py
"""

class Agent():
    def __init__(self, actions, verbose=False):
        self.actions_list = actions
        self.num_actions = len(actions)

        self.Q = {(): np.ones(self.num_actions)}

        self.verbose = verbose
        self.set_learning_options()
        self.used_actions = []
        self.powerset = None

        self.steps = 0
        self.rewards = 0
        self.total_trials = 0
        self.total_successes = 0

        self.url_first = "https://www.zixem.altervista.org/SQLi/level1.php?id=1 "
        self.url = ""

    def set_learning_options(self, exploration=0.9, learningrate=0.1, discount=0.9, max_step=100):
        self.expl = exploration
        self.lr = learningrate
        self.discount = discount
        self.max_step = max_step

    def _select_action(self, learning=True):
        if (np.random.random() < self.expl and learning):
            return np.random.randint(0, self.num_actions)
        else:
            q_value = self.Q[self.state]
            if isinstance(q_value, torch.Tensor):
                q_value = q_value.cpu().numpy()
            if q_value.ndim == 1:
                action = np.argmax(q_value)
            elif q_value.ndim == 2:
                action = np.argmax(q_value, axis=1)
            else:
                raise ValueError(f"Q-table has shape {q_value.shape}, not supported")
            return action

    def _response(self, action):
        self.url = str(self.url_first) + str(action)
        html_content = requests.get(self.url).text
        soup = BeautifulSoup(html_content, "html.parser")

        text_patterns = {
            "zixem@localhost": "zixem@localhost",
            "8.0.36": "8.0.36",
            "You have an error in your SQL syntax;": "You have an error in your SQL syntax",
            "The used SELECT statements have a different number of columns": "The used SELECT statements have a different number of columns",
            "WAF block sqli payload": "WAF block sqli payload"
        }
        for pattern, response in text_patterns.items():
            if soup.find(string=lambda s: s and pattern in s):
                return response
        return ""

    def step(self, deterministic=False):
        self.steps = self.steps + 1
        action_num = self._select_action(learning=not deterministic)
        action = self.actions_list[action_num]
        if self.verbose:
            print(action)
        self.used_actions.append(action)
        response = self._response(action)
        state_resp, reward, termination, debug_msg = self.env.step(response)

        self.rewards = self.rewards + reward
        self._analyze_response(action_num, state_resp, reward, learning=not deterministic)
        self.terminated = termination
        if self.verbose:
            print(debug_msg)

        return action_num, reward, termination

    def _update_state(self, action_nr, response_interpretation):
        action_nr += 1
        x = list(set(list(self.state) + [response_interpretation * action_nr]))
        x.sort()
        x = tuple(x)
        self.Q[x] = self.Q.get(x, np.ones(self.num_actions))
        self.oldstate = self.state
        self.state = x

    def _analyze_response(self, action, response, reward, learning=True):
        expl1 = 1  # get user
        expl2 = 2  # get version
        expl3 = 3  # syntax
        expl4 = 4  # difference col
        expl5 = 5  # waf block
        wrong1 = -1  # nothing

        if response == expl1 or response == expl2:
            self._update_state(action, response_interpretation=1)
            if learning:
                self._update_Q(action, reward)
        elif response == expl3:
            self._update_state(action, response_interpretation=-1)
            if learning:
                self._update_Q(action, reward)
        elif response == expl4:
            self._update_state(action, response_interpretation=-1)
            if learning:
                self._update_Q(action, reward)
        elif response == expl5:
            self._update_state(action, response_interpretation=-1)
            if learning:
                self._update_Q(action, reward)
        elif response == wrong1:
            self._update_state(action, response_interpretation=-1)
            if learning:
                self._update_Q(action, reward)
        else:
            print("ILLEGAL RESPONSE")
            sys.exit()

    def _update_Q(self, action, reward):
        q_value = self.Q[self.state]
        if isinstance(q_value, torch.Tensor):
            q_value = q_value.cpu().numpy()
        best_action_newstate = np.argmax(q_value)

        self.Q[self.oldstate][action] = self.Q[self.oldstate][action] + self.lr * (
            reward + self.discount * self.Q[self.state][best_action_newstate] - self.Q[self.oldstate][action]
        )

    def reset(self, env):
        self.env = env
        self.terminated = False
        self.state = ()
        self.oldstate = None
        self.used_actions = []
        self.steps = 0
        self.rewards = 0

    def run_episode(self, deterministic=False):
        _, _, self.terminated, s = self.env.reset()
        if self.verbose:
            print(s)

        episode_actions = []
        episode_rewards = 0
        episode_steps = 0

        while not self.terminated and self.steps < self.max_step:
            action_num, reward, termination = self.step(deterministic=deterministic)
            episode_actions.append(self.actions_list[action_num])
            episode_rewards += reward
            episode_steps += 1
            self.terminated = termination

        self.total_trials += 1
        if self.terminated:
            self.total_successes += 1

        return self.terminated, episode_steps, episode_rewards, episode_actions

    def run_human_look_episode(self):
        _, _, self.terminated, s = self.env.reset()
        print(s)
        while not self.terminated and self.steps < self.max_step:
            self.look_step()
        self.total_trials += 1
        if self.terminated:
            self.total_successes += 1
        return self.terminated

    def look_step(self):
        self.steps = self.steps + 1
        print("step", self.steps)
        print("My state is")
        print(self.state)
        print("My Q row looks like this:")
        print(self.Q[self.state])
        print("Action ranking is")
        print(np.argsort(self.Q[self.state])[::-1])
        action = self._select_action(learning=True)
        print("action equal highest rank", action == np.argsort(self.Q[self.state])[::-1][0])

        state_resp, reward, termination, debug_msg = self.env.step(action)
        self._analyze_response(action, state_resp, reward)
        self.terminated = termination
        print(debug_msg)

if __name__ == "__main__":
    a = Agent(const.actions)
    env = srv.mockSQLenv()
    a.reset(env)
    a.run_episode()