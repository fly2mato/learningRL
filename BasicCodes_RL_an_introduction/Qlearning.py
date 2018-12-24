import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from RLalgorithm import RLalgorithm

Action = ['R', 'L', 'U', 'D']

class Qlearning(RLalgorithm):
    def __init__(self, state_size, action_size, epsilon=0.1, gamma=1, alpha=0.5):
        super(Qlearning, self).__init__(state_size, action_size)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha #learning rate
        self.episode_steps = []

    def plot_steps_per_episode(self):
        plt.plot(np.arange(1, len(self.episode_steps)-1), self.episode_steps[2:])
        plt.xlabel('Episodes')
        plt.ylabel('stpes')
        # plt.ylim(0, 500)
        plt.show()
        

    def choose_action(self, observation):
        if np.random.binomial(1, self.epsilon) == 0:
            return self.choose_action_best(observation)
        else:
            return np.random.choice(self.action_size)
    
    def choose_action_best(self, observation):
        x,y = observation
        index = np.random.permutation(self.action_size)
        a = index[np.argmax(self.Q[x,y][index])]
        return a

    def learn(self, observation, action, observation_, reward):
        action_ = self.choose_action_best(observation_)

        S_A = tuple(observation + [action])
        S_A_ = tuple(observation_ + [action_])

        self.Q[S_A] += self.alpha * (reward + self.gamma * self.Q[S_A_] - self.Q[S_A])

    def train(self, max_episode, env):
        count = 0
        for _ in tqdm(range(max_episode)):
            reward_sum = 0
            steps = 0
            for _ in range(10):
                notTerminal = 1
                observation = env.reset()
                while(notTerminal):# and count < 100):
                    count += 1
                    steps += 1
                    action = self.choose_action(observation) #初始动作
                    observation_, reward, notTerminal = env.step(action)
                    self.learn(observation, action, observation_, reward)
                    observation = observation_
                    reward_sum += reward
            
            self.episode_steps.append(steps/10)
            self.episode_end_time.append(count)
            self.sum_reward.append(reward_sum/10)

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