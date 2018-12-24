import numpy as np
from tqdm import tqdm

from RLalgorithm import RLalgorithm
from Qlearning import Qlearning

Action = ['R', 'L', 'U', 'D']

class Dyna_Q(Qlearning):
    def __init__(self, state_size, action_size, epsilon=0.1, gamma=1.0, alpha=0.5, sim_n=100, plus=False):
        super(Dyna_Q, self).__init__(state_size, action_size)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha #learning rate
        self.simulation_times = sim_n
        self.model = dict()
        self.plus = plus

    def modeling(self, observation, action, observation_, reward):
        # if tuple(observation+[action]) not in self.model:
        self.model[tuple(observation+[action])] = observation_+[reward]

        if self.plus:
            for action_ in range(self.action_size):
                if tuple(observation+[action_]) not in self.model:
                    self.model[tuple(observation+[action_])] = observation+[0] 

        for _ in range(self.simulation_times):
            x,y,a = list(self.model.keys())[np.random.choice(len(self.model.keys()))]
            x_,y_,reward_sim = self.model[(x,y,a)]
            self.learn([x,y], a, [x_,y_], reward_sim)

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
                    self.modeling(observation, action, observation_, reward)
                    observation = observation_
                    reward_sum += reward

            self.episode_steps.append(steps/10)
            self.episode_end_time.append(count)
            self.sum_reward.append(reward_sum/10)

