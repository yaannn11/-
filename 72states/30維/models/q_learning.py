import numpy as np

class QLearningAgent:
    def __init__(self,num_actions=4,num_states=8,alpha=0.1,gamma=0.9,T=1.0):

        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma

        # Temperature上下限
        self.T = T
        self.T_min = 0.3
        self.T_max = 2.5

        self.q_table = np.full((num_states, num_actions), 0.25)

        # 紀錄Temperature (之後畫圖)
        self.temperature_history = []
        
    def select_action(self, state):
        q_values = self.q_table[state]
        exp_q = np.exp((q_values - np.max(q_values)) / self.T)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(self.num_actions, p=probs)
    
    def update_temperature(self, improvement_rate):
        self.T = self.T_max - improvement_rate * (self.T_max - self.T_min)
        self.T = np.clip(self.T,self.T_min,self.T_max)
        self.temperature_history.append(self.T)
        
    def update(self, state, action, reward, next_state):
        best_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])
