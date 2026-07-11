class QLearningAgent:
    def __init__(self, num_actions=4, num_states=8, alpha=0.1, gamma=0.9, T=1.0):
        self.num_actions = num_actions
        self.num_states = num_states
        self.alpha = alpha
        self.gamma = gamma
        self.T = T
        self.q_table = np.full((num_states, num_actions), 0.25)
        
    def select_action(self, state):
        q_values = self.q_table[state]
        exp_q = np.exp((q_values - np.max(q_values)) / self.T)
        probs = exp_q / np.sum(exp_q)
        return np.random.choice(self.num_actions, p=probs)
        
    def update(self, state, action, reward, next_state):
        best_next_q = np.max(self.q_table[next_state])
        td_target = reward + self.gamma * best_next_q
        self.q_table[state, action] += self.alpha * (td_target - self.q_table[state, action])
