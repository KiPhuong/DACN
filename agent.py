import numpy as np
import mockSQLenv as srv
import const
import sys
import utilities as ut

import requests
from bs4 import BeautifulSoup


"""
agent.py is based on FMZennaro's agent on https://github.com/FMZennaro/CTF-RL/blob/master/Simulation1/agent.py
"""

class Agent():
	def __init__(self, actions, verbose=True):
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

		#self.url_first = "http://localhost/2291d7161c845409ba5a9c13ef489d0d5d345f6a.php?input="
		self.url_first = "https://www.zixem.altervista.org/SQLi/level1.php?id=1 "
		self.url =""

	def set_learning_options(self,exploration=0.9,learningrate=0.1,discount=0.9, max_step = 100):
		self.expl = exploration
		self.lr = learningrate
		self.discount = discount
		self.max_step = max_step

	def _select_action(self, learning = True):
		if (np.random.random() < self.expl and learning):
			return np.random.randint(0,self.num_actions)
		else:
			return np.argmax(self.Q[self.state])


	# def _response (self, action):
	# 	self.url = str(self.url_first) + str(action)
	# 	print(self.url)
	# 	html_content = requests.get(self.url).text
	# 	soup = BeautifulSoup(html_content, "html.parser")
	# 	version_text = soup.find(string=lambda s: s and "zixem@localhost" in s)
	# 	user_text = soup.find(string=lambda s: s and "8.0.36" in s)
	# 	error_syntax = soup.find(string=lambda s: s and "You have an error in your SQL syntax;" in s)
	# 	error_diff_collumn = soup.find(string=lambda s: s and "The used SELECT statements have a different number of columns" in s)
	# 	error_waf_block = soup.find(string=lambda s: s and "WAF block sqli payload" in s)
	# 	text = ["zixem@localhost", "8.0.36", "syntax","different number of columns", "WAF block sqli payload" ]

	# 	reponse =""
	# 	if (version_text != None): 
	# 		reponse = "zixem@localhost"
	# 		return reponse
	# 	elif (user_text != None):
	# 		reponse = "8.0.36"
	# 		return reponse
	# 	elif (error_syntax != None):
	# 		reponse = "You have error in your SQL syntax"
	# 		return reponse
	# 	elif (error_diff_collumn != None):
	# 		reponse = "The used SELECT statements have a different number of columns"
	# 		return reponse
	# 	elif (error_waf_block != None):
	# 		reponse = "WAF block sqli payload"
	# 		return reponse
	# 	else: return reponse

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
		#return nothing
		return ""	

	def step(self, deterministic = False):
		self.steps = self.steps + 1
		action_num = self._select_action(learning = not deterministic)
		action = self.actions_list[action_num]
		if (self.verbose): print(action)
		self.used_actions.append(action)
		reponse = self._response(action)
		state_resp, reward, termination, debug_msg = self.env.step(reponse)

		
		#print(state_resp)
		#print(reward)
		#print(termination)
		#print(debug_msg)

		self.rewards = self.rewards + reward

		self._analyze_response(action_num, state_resp, reward, learning = not deterministic)
		self.terminated = termination
		#if(self.terminated): print("Done episode")
		if(self.verbose): print(debug_msg)

		return


	def _update_state(self, action_nr, response_interpretation):
		"""
		response interpretation is either -1 or 1
		"""
		action_nr += 1
		x = list(set(list(self.state) + [response_interpretation*action_nr]))
		x.sort()
		x = tuple(x)
		self.Q[x] = self.Q.get(x, np.ones(self.num_actions))

		self.oldstate = self.state
		self.state = x




	def _analyze_response(self, action, response, reward, learning = True):
		expl1 = 1 	#get user
		expl2 = 2 	#get version
		expl3  = 3 	#syntax
		expl4 = 4 	#difference col
		expl5 = 5 # waf block
		wrong1 = -1 	#nothing

		#The agent recieves data as the response
		if(response==expl1 or response == expl2):
			self._update_state(action, response_interpretation = 1)
			if(learning): self._update_Q(action, reward)
		#syntax
		elif(response == expl3):
			self._update_state(action, response_interpretation = -1)
			if(learning): self._update_Q(action, reward)
		#differ col
		elif(response==expl4):
			self._update_state(action, response_interpretation = -1)
			if(learning): self._update_Q(action, reward)
		#waf block
		elif(response==expl5):
			self._update_state(action, response_interpretation = -1)
			if(learning): self._update_Q(action, reward)
		#nothin
		elif(response==wrong1):
			self._update_state(action, response_interpretation = -1)
			if(learning): self._update_Q(action,reward)
		else:
			print("ILLEGAL RESPONSE")
			sys.exit()

	def _update_Q(self, action, reward):
		best_action_newstate = np.argmax(self.Q[self.state])
		self.Q[self.oldstate][action] = self.Q[self.oldstate][action] + self.lr * (reward + self.discount*self.Q[self.state][best_action_newstate] - self.Q[self.oldstate][action])

	def reset(self,env):
		self.env = env
		self.terminated = False
		self.state = () #empty tuple
		self.oldstate = None
		self.used_actions = []

		self.steps = 0
		self.rewards = 0


	def run_episode(self, deterministic = False):
		_,_,self.terminated,s = self.env.reset()

		if(self.verbose): print(s)

		#Limiting the maximimun number of steps we allow the attacker to make to avoid overly long runtimes and extreme action spaces
		while (not(self.terminated)) and self.steps < self.max_step:
			self.step(deterministic = deterministic)


		self.total_trials += 1
		if(self.terminated):
			self.total_successes += 1
		return self.terminated

	def run_human_look_episode(self):
		_,_,self.terminated,s = self.env.reset()
		print(s)
		while (not(self.terminated)) and self.steps < self.max_step:
			self.look_step()
		self.total_trials += 1
		if(self.terminated):
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
		action = self._select_action(learning = True)

		print("action equal highest rank",action == np.argsort(self.Q[self.state])[::-1][0])


		state_resp, reward, termination, debug_msg = self.env.step(action)
		self._analyze_response(action, state_resp, reward)
		self.terminated = termination
		print(debug_msg)



if __name__ == "__main__":
	a = Agent(const.actions)
	env = srv.mockSQLenv()
	a.reset(env)
	#a.run_human_look_episode()
	a.run_episode()
	#a.step()
	#print(a.Q)
	#a.step()
