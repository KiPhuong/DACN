
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from bs4 import BeautifulSoup

from env import mockSQLenv as srv
from env import const
from env import utilities as ut

torch.cuda.empty_cache()


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super(DQN, self).__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)

    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)


class Agent():
    def __init__(self, actions, verbose=True, exploration=1.0):
        self.actions_list = actions
        self.num_actions = len(actions)
        self.verbose = verbose
        self.set_learning_options()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.state_dim = 6  # 6 loại phản hồi: user, version, syntax error, diff columns, WAF, no response
        self.model = DQN(self.state_dim, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.exploration = exploration
        self.min_exploration = 0.1
        self.decay = 0.99995

        self.action_taken = []
        self.env = srv.mockSQLenv()
        self.steps = 0
        self.rewards = 0
        self.total_trials = 0
        self.total_successes = 0

        self.url_first = "https://www.zixem.altervista.org/SQLi/level1.php?id=1 "
        self.url = ""
        self.last_response = ""

    def set_learning_options(self, discount=0.9, max_step=100):
        self.discount = discount
        self.max_step = max_step

    def encode_state(self, response):
        # One-hot encode phản hồi
        state = torch.zeros(self.state_dim, dtype=torch.float32, device=self.device)
        response_map = {
            "zixem@localhost": 0,
            "8.0.36": 1,
            "You have an error in your SQL syntax": 2,
            "The used SELECT statements have a different number of columns": 3,
            "WAF block sqli payload": 4,
            "": 5  # no response
        }
        index = response_map.get(response, 5)
        state[index] = 1.0
        return state

    def _select_action(self, state, learning=True):
        if np.random.rand() < self.exploration and learning:
            return np.random.randint(0, self.num_actions)
        else:
            with torch.no_grad():
                q_values = self.model(state.unsqueeze(0))
                return torch.argmax(q_values).item()

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

    def step(self, env):
        self.steps += 1
        state = self.encode_state(self.last_response)
        action_num = self._select_action(state)
        action = self.actions_list[action_num]
        self.action_taken.append(action)

        response = self._response(action)
        state_resp, reward, terminated, debug_msg = self.env.step(response)

        if self.verbose:
            print("response =", response)
            print("state response =", state_resp)
            print("reward =", reward)
            print("terminated:", terminated)
            print("message =", debug_msg)

        self.rewards += reward
        next_state = self.encode_state(response)

        with torch.no_grad():
            q_target = reward + (self.discount * torch.max(self.model(next_state.unsqueeze(0))) * (1 - terminated))

        q_values = self.model(state.unsqueeze(0))
        loss = self.loss_fn(q_values[0, action_num], q_target)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        if self.verbose:
            print(f"Step {self.steps} | Action: {action} | Loss: {loss.item():.4f} | Reward: {reward}")

        if terminated:
            self.exploration = max(self.min_exploration, self.exploration * self.decay)
            self.terminated = True

        self.last_response = response

    def reset(self, env):
        self.env = env
        self.terminated = False
        self.last_response = ""
        self.state = ()
        self.oldstate = None
        self.used_actions = []
        self.steps = 0
        self.rewards = 0

    def run_episode(self, env):
        self.env = env
        _, _, self.terminated, s = self.env.reset()

        while not self.terminated and self.steps < self.max_step:
            self.step(env)

        if self.verbose:
            print(self.steps)
            print(self.rewards)
            print(self.model)
            for name, param in self.model.state_dict().items():
                print(name, param.shape)

        self.total_trials += 1
        if self.terminated:
            self.total_successes += 1

        return self.terminated, self.steps, self.rewards, self.action_taken


if __name__ == "__main__":
    a = Agent(const.actions, verbose=True)
    env = srv.mockSQLenv()
    a.reset(env)
    a.run_episode(env)
