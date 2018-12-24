import numpy as np
import matplotlib.pyplot as plt

class RLalgorithm(object):
    def __init__(self, state_size=[1,1], action_size=4):
        self.state_size = state_size
        self.action_size = action_size
        self.V = np.zeros(state_size)
        self.Q = np.zeros(state_size+[action_size])
        self.episode_end_time = [0]
        self.sum_reward = []

    def choose_action(self, observation):
        a = np.random.choice([0,1,2,3])
        return a

    def learn(self, observation, action, observation_, reward):
        pass

    def plot_episode_time(self):
        plt.plot(self.episode_end_time, np.arange(1, len(self.episode_end_time) + 1))
        plt.scatter(self.episode_end_time[:20:], np.arange(1, len(self.episode_end_time) + 1)[:20:], marker='s', s=10, c='r')
        plt.xlabel('Time steps')
        plt.ylabel('Episodes')
        plt.show()
    
    def plot_sumreward_episode(self):
        plt.plot(np.arange(1, len(self.sum_reward) + 1), self.sum_reward)
        plt.xlabel('Episodes')
        plt.ylabel('Sum of rewards')
        plt.show()