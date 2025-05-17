import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import requests
from bs4 import BeautifulSoup

from env import mockSQLenv as srv
from env import const
from env import utilities as ut

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
        self.num_actions = len(actions)  # 100 actions
        self.verbose = verbose
        self.set_learning_options()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.history_length = 1  # Lưu 1 hành động gần nhất
        self.num_states = 6  # 6 loại phản hồi từ server
        self.input_dim = self.num_states + self.history_length * self.num_actions  # 6 + 1 * 100 = 106
        self.model = DQN(self.input_dim, self.num_actions).to(self.device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=0.001)
        self.loss_fn = nn.MSELoss()
        self.exploration = exploration
        self.min_exploration = 0.1
        self.decay = 0.99995

        self.action_used = []  # Lưu toàn bộ hành động trong 1 tập train
        self.action_taken = []  # Lưu hành động vừa lấy được
        self.loss_history = []  # Lưu lịch sử loss
        self.last_response_idx = None  # Lưu response_idx của bước trước
        self.env = srv.mockSQLenv()
        self.steps = 0
        self.rewards = 0
        self.total_trials = 0
        self.total_successes = 0

        self.url_first = "http://www.zixem.altervista.org/SQLi/level1.php?id=1"
        self.url = ""

    def set_learning_options(self, discount=0.9, max_step=100):
        self.discount = discount
        self.max_step = max_step

    def _select_action(self, state, learning=True):
        if learning and np.random.rand() < self.exploration:
            # Chính sách dựa trên trạng thái
            action_probs = torch.ones(self.num_actions, device=self.device) / self.num_actions  # Phân phối mặc định
            
            # Kiểm tra state hợp lệ
            if state is None or state.shape[0] != self.input_dim:
                if self.verbose:
                    print(f"Invalid state: {state}")
                response_idx = 0  # Giá trị mặc định nếu state không hợp lệ
            else:
                # Lấy phản hồi từ trạng thái (6 chiều đầu tiên)
                response_vector = state[:self.num_states]
                response_idx = 0  # Giá trị mặc định
                try:
                    if response_vector.sum() > 0:
                        response_idx = torch.argmax(response_vector).item()  # Cập nhật nếu có phản hồi
                except Exception as e:
                    if self.verbose:
                        print(f"Error in calculating response_idx: {e}, response_vector: {response_vector}")
                    response_idx = 0  # Fallback

            # Gán xác suất dựa trên phản hồi
            if response_idx == 0:  # -1: Không có phản hồi
                if self.last_response_idx in [3, 4]:  # Bước trước là lỗi cú pháp hoặc cột
                    action_probs[30:40] *= 3.0  # Payload đúng Level 1
                    action_probs[40:70] *= 2.0  # Payload lấy dữ liệu khác
                    action_probs[12:30] *= 0.5  # Giảm kiểm tra số cột
                else:  # Bước trước là -1 hoặc None
                    action_probs[0:12] *= 2.0   # Kiểm tra lỗ hổng (0-11)
                    action_probs[12:30] *= 1.5  # Kiểm tra số cột (12-29)
                action_probs /= action_probs.sum()
            elif response_idx in [1, 2]:  # 1, 2: Thành công, ưu tiên khai thác
                with torch.no_grad():
                    q_values = self.model(state.unsqueeze(0))
                    action_probs = torch.softmax(q_values, dim=1).squeeze()
            elif response_idx == 3:  # 3: Lỗi cú pháp, ưu tiên lấy dữ liệu
                action_probs[30:40] *= 3.0  # Payload đúng Level 1
                action_probs[40:70] *= 2.0  # Payload lấy dữ liệu khác
                action_probs[12:30] *= 1.5  # Kiểm tra số cột
                # Nếu bước trước cũng là lỗi cú pháp, tăng ưu tiên payload đúng
                if self.last_response_idx == 3:
                    action_probs[30:40] *= 4.0  # Tăng mạnh ưu tiên
                    action_probs[12:30] *= 0.5  # Giảm kiểm tra số cột
                action_probs /= action_probs.sum()
            elif response_idx == 4:  # 4: Lỗi số cột, ưu tiên lấy dữ liệu 3 cột
                action_probs[30:40] *= 3.0  # Payload đúng Level 1
                action_probs[40:70] *= 2.0  # Payload lấy dữ liệu khác
                action_probs[18:21] *= 2.0  # union select null,null,null (~18-20)
                # Nếu bước trước cũng là lỗi cột, tăng ưu tiên payload đúng
                if self.last_response_idx == 4:
                    action_probs[30:40] *= 4.0  # Tăng mạnh ưu tiên
                    action_probs[12:30] *= 0.5  # Giảm kiểm tra số cột
                action_probs /= action_probs.sum()
            elif response_idx == 5:  # 5: WAF block (không áp dụng Level 1)
                action_probs[30:70] *= 2.0  # Fallback lấy dữ liệu
                action_probs /= action_probs.sum()

            # Giảm xác suất cho action không hiệu quả
            if response_idx == 0 and self.action_taken:
                last_action = self.action_taken[-1]
                action_probs[last_action] *= 0.5  # Giảm xác suất
                action_probs /= action_probs.sum()

            # Lấy mẫu hành động từ phân phối
            action_num = torch.multinomial(action_probs, 1).item()
        else:
            # Khai thác: Chọn hành động có Q-value cao nhất
            with torch.no_grad():
                if state is None or state.shape[0] != self.input_dim:
                    if self.verbose:
                        print(f"Invalid state in exploitation: {state}")
                    action_num = np.random.randint(self.num_actions)  # Fallback
                else:
                    q_values = self.model(state.unsqueeze(0))
                    action_num = torch.argmax(q_values).item()
        
        # Cập nhật last_response_idx
        self.last_response_idx = response_idx
        return action_num

    def _response(self, action):
        self.url = f"{self.url_first}{action}"
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

    def _get_state(self, state_resp=None):
        # Mã hóa phản hồi
        resp_vector = torch.zeros(self.num_states, dtype=torch.float32, device=self.device)
        if state_resp is not None:
            resp_index = {-1: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5}[state_resp]
            resp_vector[resp_index] = 1.0
        
        # Mã hóa lịch sử hành động (chỉ 1 hành động)
        history_vector = torch.zeros(self.history_length * self.num_actions, dtype=torch.float32, device=self.device)
        if self.action_taken:
            action_idx = self.action_taken[-1]
            history_vector[action_idx] = 1.0
        
        # Kết hợp và đảm bảo tensor hợp lệ
        state = torch.cat([resp_vector, history_vector])
        if state.shape[0] != self.input_dim:
            if self.verbose:
                print(f"Invalid state shape: {state.shape}, expected: {self.input_dim}")
            state = torch.zeros(self.input_dim, dtype=torch.float32, device=self.device)
        return state

    def _update_history(self, action_idx):
        self.action_taken = [action_idx]  # Chỉ giữ 1 hành động

    def _check_model_update(self):
        """Kiểm tra sự thay đổi của trọng số và gradient của mô hình."""
        if self.verbose:
            print("\n--- Model Update Check ---")
            for name, param in self.model.named_parameters():
                if param.grad is not None:
                    mean_weight = param.data.mean().item()
                    std_weight = param.data.std().item()
                    mean_grad = param.grad.mean().item()
                    std_grad = param.grad.std().item()
                    print(f"Layer {name}:")
                    print(f"  Weight mean: {mean_weight:.6f}, std: {std_weight:.6f}")
                    print(f"  Gradient mean: {mean_grad:.6f}, std: {std_grad:.6f}")
            if self.loss_history:
                avg_loss = np.mean(self.loss_history)
                print(f"Average loss in episode: {avg_loss:.6f}")

    def step(self, env):
        self.steps += 1
        
        # Lấy trạng thái hiện tại
        state = self._get_state()
        
        # Chọn hành động
        action_num = self._select_action(state)
        action = self.actions_list[action_num]
        self.action_used.append(action)  # Ghi action vào list
        self._update_history(action_num)  # Cập nhật lịch sử
        
        # Thực hiện hành động
        response = self._response(action)
        state_resp, reward, terminated, debug_msg = self.env.step(response)
        
        # Lấy trạng thái tiếp theo
        next_state = self._get_state(state_resp)
        
        if self.verbose:
            print("response =", response)
            print("state response =", state_resp)
            print("reward =", reward)
            print("terminated:", terminated)
            print("message =", debug_msg)
        
        self.rewards += reward
        
        # Tính loss
        with torch.no_grad():
            q_target = reward + (self.discount * torch.max(self.model(next_state.unsqueeze(0))) * (1 - terminated))
        
        q_values = self.model(state.unsqueeze(0))
        loss = self.loss_fn(q_values[0, action_num], q_target)
        self.loss_history.append(loss.item())  # Lưu loss

        self.optimizer.zero_grad()
        loss.backward()
        if self.verbose:
            print(f"Step {self.steps} | Action: {action} | Loss: {loss.item():.4f} | Reward: {reward}")

        self.optimizer.step()

        if terminated:
            self.exploration = max(self.min_exploration, self.exploration * self.decay)
            self.terminated = True

    def reset(self, env):
        self.env = env
        self.terminated = False
        self.action_taken = []
        self.action_used = []
        self.last_response_idx = None  # Reset response_idx
        self.loss_history = []  # Xóa lịch sử loss
        self.steps = 0
        self.rewards = 0

    def run_episode(self, env):
        self.env = env
        _, _, self.terminated, s = self.env.reset()
        
        while (not self.terminated) and self.steps < self.max_step:
            self.step(env)

        if self.verbose:
            print("Steps:", self.steps)
            print("Rewards:", self.rewards)
            print("Model:", self.model)
            for name, param in self.model.state_dict().items():
                print(name, param.shape)
            self._check_model_update()  # Kiểm tra cập nhật mô hình

        self.total_trials += 1
        if self.terminated:
            self.total_successes += 1
        
        return self.terminated, self.steps, self.rewards, self.action_taken, self.action_used

if __name__ == "__main__":
    a = Agent(const.actions, verbose=True)
    env = srv.mockSQLenv()
    a.reset(env)
    a.run_episode(env)