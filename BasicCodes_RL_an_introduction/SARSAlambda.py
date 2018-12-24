import numpy as np
from tqdm import tqdm

from RLalgorithm import RLalgorithm

Action = ['R', 'L', 'U', 'D']

class SARSAlambda(RLalgorithm):
    def __init__(self, state_size, action_size, epsilon=0.1, gamma=1, alpha=0.5, slambda = 0.9):
        super(SARSAlambda, self).__init__(state_size, action_size)
        self.epsilon = epsilon
        self.gamma = gamma
        self.alpha = alpha #learning rate
        self.Eligibility = np.zeros(state_size+[action_size])
        self.slambda = slambda

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

    def learn(self, observation, action, observation_, action_, reward):
        S_A = tuple(observation + [action])
        S_A_ = tuple(observation_ + [action_])
        self.Eligibility[tuple(observation)] = 0
        self.Eligibility[S_A] += 1
        # self.Eligibility[self.Eligibility>1]=1
        delta = reward + self.gamma * self.Q[S_A_] - self.Q[S_A] 
        self.Q += self.alpha * delta * self.Eligibility
        self.Eligibility *= self.gamma * self.slambda

    def train(self, max_episode, env):
        count = 0
        for _ in tqdm(range(max_episode)):
            reward_sum = 0
            for _ in range(1):
                notTerminal = 1
                observation = env.reset()
                action = self.choose_action(observation) #初始动作
                self.Eligibility = np.zeros(self.state_size+[self.action_size])
                while(notTerminal):# and count < 100):
                    count += 1
                    observation_, reward, notTerminal = env.step(action)
                    action_ = self.choose_action(observation_)
                    self.learn(observation, action, observation_, action_, reward)
                    observation = observation_
                    action = action_
                    reward_sum += reward

            self.episode_end_time.append(count)
            self.sum_reward.append(reward_sum/1)

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