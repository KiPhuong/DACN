import numpy as np
import const

class mockSQLenv(object):
    """
    """
    def __init__(self, verbose=False, get_uname_rw=10, get_version_rw = 5 ,syntax_reward=-2, differ_col_reward=-1, query_reward=-3, waf_block=-20):
        # Get the action space
        self.A = np.array(const.actions)
        self.query_reward = query_reward #nothing ressponse
        self.get_uname_rw = get_uname_rw #get data
        self.get_version_rw = get_version_rw
        self.syntax_reward = syntax_reward # syntax error
        self.differ_col_reward = differ_col_reward
        self.waf_block = waf_block
        self.verbose = verbose
        self.termination = False
        self.debug_msg = ""  # Khởi tạo thuộc tính debug_msg


    def step(self, response):
        # Process action
        if response == "zixem@localhost":
            if self.verbose:
                print('Correct exploratory user of the challenge. I return 1')
            self.termination = True
            self.debug_msg = 'Server response is 1'  # Lưu thông điệp vào debug_msg
            return 1, self.get_uname_rw, self.termination, self.debug_msg
        elif response == "8.0.36":
            if self.verbose:
                print('Correct exploratory version of the escape. I return 2')
            self.termination = True
            self.debug_msg = 'Server response is 2'  # Lưu thông điệp
            return 2, self.get_version_rw, self.termination, self.debug_msg
        elif response == "You have an error in your SQL syntax":
            if self.verbose:
                print('Syntax error. Server response 3')
            self.debug_msg = 'Server response is 3'  # Lưu thông điệp
            return 3, self.syntax_reward, self.termination, self.debug_msg
        elif response == "The used SELECT statements have a different number of columns":
            if self.verbose:
                print('Query with different number of columns. Server response 4')
            self.debug_msg = 'Server response is 4'  # Lưu thông điệp
            return 4, self.differ_col_reward, self.termination, self.debug_msg
        elif response == "WAF block sqli payload":
            if self.verbose:
                print('WAF block sqli payload. Server response 5')
            self.debug_msg = 'Server response is 5'  # Lưu thông điệp
            return 5, self.waf_block, self.termination, self.debug_msg
        else:
            if self.verbose:
                print('Nothing response')
            self.debug_msg = 'Server response is -1'  # Lưu thông điệp
            return -1, self.query_reward, self.termination, self.debug_msg

    def reset(self):
        self.termination = False
        self.debug_msg = 'Env reset'  # Lưu thông điệp khi reset
        if self.verbose:
            print('Reset env')
        return None, 0, self.termination, self.debug_msg