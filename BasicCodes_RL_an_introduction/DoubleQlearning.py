import numpy as np
from tqdm import tqdm

from RLalgorithm import RLalgorithm

Action = ['R', 'L', 'U', 'D']

class DoubleQlearning(RLalgorithm):
    def __init__(self, state_size, action_size, epsilon=0.1, gamma=1, alpha=0.5):
        self.state_size = state_size
        self.action_size = action_size
        self.Q1 = np.zeros(state_size+[action_size])
        self.Q2 = np.zeros(state_size+[action_size])
        self.episode_end_time = [0]
        self.sum_reward = []
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha #learning rate

    def choose_action(self, observation):
        if np.random.binomial(1, self.epsilon) == 0:
            return self.choose_action_best(observation)
        else:
            return np.random.choice(self.action_size)
    
    def choose_action_best(self, observation, which=0):
        x,y = observation
        index = np.random.permutation(self.action_size)
        if which==1:
            return index[np.argmax(self.Q1[x,y][index])]
        if which==2:
            return index[np.argmax(self.Q2[x,y][index])]
        return index[np.argmax(self.Q1[x,y][index] + self.Q2[x,y][index])]

    def learn(self, observation, action, observation_, reward):
        S_A = tuple(observation + [action])
        
        choice = np.random.binomial(1, 0.5)
        if choice:
            action_ = self.choose_action_best(observation_, 1)
            S_A_ = tuple(observation_ + [action_])
            self.Q1[S_A] += self.alpha * (reward + self.gamma * self.Q2[S_A_] - self.Q1[S_A])
        else:
            action_ = self.choose_action_best(observation_, 2)
            S_A_ = tuple(observation_ + [action_])
            self.Q2[S_A] += self.alpha * (reward + self.gamma * self.Q1[S_A_] - self.Q2[S_A])

    def train(self, max_episode, env):
        count = 0
        for _ in tqdm(range(max_episode)):
            reward_sum = 0
            for _ in range(50):
                notTerminal = 1
                observation = env.reset()
                while(notTerminal):# and count < 100):
                    count += 1
                    action = self.choose_action(observation) #初始动作
                    observation_, reward, notTerminal = env.step(action)
                    self.learn(observation, action, observation_, reward)
                    observation = observation_
                    reward_sum += reward

            self.episode_end_time.append(count)
            self.sum_reward.append(reward_sum/50)

    def run(self, env):
        notTerminal = 1
        observation = env.reset()
        count = 0
        while(notTerminal):
            count += 1
            action = self.choose_action(observation)
            print(observation, Action[action], count)
            observation_, _ , notTerminal = env.step(action)
            observation = observation_