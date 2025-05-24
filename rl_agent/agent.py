import numpy as np
import random

class SleepRLAgent:
    def __init__(self, stages=['Light', 'Deep', 'REM'], actions=['Play', 'Wait'], alpha=0.1, gamma=0.9, epsilon=0.2):
        self.states = stages
        self.actions = actions
        self.q_table = np.zeros((len(stages), len(actions)))
        self.alpha = alpha  # learning rate
        self.gamma = gamma  # discount factor
        self.epsilon = epsilon  # exploration rate

    def get_action(self, state_idx):
        if random.uniform(0, 1) < self.epsilon:
            return random.randint(0, len(self.actions) - 1)  # Explore
        return np.argmax(self.q_table[state_idx])  # Exploit

    def update(self, state_idx, action_idx, reward):
        predict = self.q_table[state_idx, action_idx]
        target = reward  # No next state considered here (1-step Q-learning)
        self.q_table[state_idx, action_idx] += self.alpha * (target - predict)
