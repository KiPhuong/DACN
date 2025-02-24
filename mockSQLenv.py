import numpy as np

import const

class mockSQLenv(object):
	"""
	"""
	def __init__(self,verbose=False, data_reward = 10, syntax_reward = -5, differ_col_reward = -1, query_reward = -2, waf_block = -20):
		# Get the action space
		self.A = np.array(const.actions)
		self.query_reward = query_reward
		self.data_reward = data_reward
		self.syntax_reward = syntax_reward
		self.differ_col_reward = differ_col_reward
		self.waf_block = waf_block
		self.verbose = verbose
		self.termination = False

		url_first = "https://zixem.altervista.org/SQLi/level1.php?id=1 "



	def step(self, reponse):
		# Process action
		if (reponse== "zixem@localhost"):
			if(self.verbose): print('Correct exploratory user of the challenge. I return 1')
			self.termination = True
			return 1,self.data_reward,self.termination,'Server response is 1'
		elif (reponse == "8.0.36"):
			if(self.verbose): print('Correct exploratory version of the escape. I return 2')
			self.termination = True
			return 2,self.data_reward,self.termination,'Server response is 2'
		elif (reponse=="You have an error in your SQL syntax"):
			if(self.verbose): print('Syntax error. Server response 3')
			return 3,self.syntax_reward,self.termination,'Server response is 3'
		elif (reponse == "The used SELECT statements have a different number of columns"):
			if(self.verbose): print('Query with differenc number of column. Server reponse 4')
			return 4,self.differ_col_reward, self.termination, "Server response is 4"
		elif (reponse == "WAF block sqli payload"):
			if(self.verbose): print('WAF block sqli payload. Server respone 5')
			return 5,self.waf_block, self.termination, "Server response is 5"
		else:
			if(self.verbose): print('Nothing reponse')
			return -1,self.query_reward, self.termination,'Server response is -1'


	def reset(self):
		self.termination = False
		#if(self.verbose): print('Reset env')
		return None,0,self.termination,'Env reset'
