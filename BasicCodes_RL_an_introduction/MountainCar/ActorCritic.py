from TileCodingSoftware import *

import numpy as np

class Critic(object):
    def __init__(self, alpha=0.5, gamma = 0.95):
        self.alpha = alpha/8
        self.gamma = gamma
        self.hash_size = 2048
        self.iht = IHT(self.hash_size)
        self.weight = np.zeros(self.hash_size)
        self.gamma_pow = 1

    def reset(self):
        self.gamma_pow = 1

    def get_actives(self, state):
        return tiles(self.iht, 8, [8*state[0]/(0.6+1.2), 8*state[1]/(0.07+0.07)])

    def vvalue_function(self, state):
        if state[0] >= 0.6:
            return 0
        actives = self.get_actives(state)
        return np.sum(self.weight[actives])
    
    def learn(self, observation, reward, observation_):
        actives = self.get_actives(observation)
        delta = reward + self.gamma * self.vvalue_function(observation_) \
                            - self.vvalue_function(observation)
        for a in actives:
            self.weight[a] += self.alpha * delta * self.gamma_pow

        self.gamma_pow *= self.gamma

        return delta


class Actor(object):
    def __init__(self, epsilon=0.05, actions_num=3, alpha=0.5, gamma=0.95):
        self.actions = np.arange(actions_num, dtype=np.int)
        self.epsilon = epsilon
        self.alpha = alpha
        self.gamma = gamma

        self.hash_size = 4096
        self.iht = IHT(self.hash_size)
        self.weight = np.zeros(self.hash_size)

    def get_actives(self, state, action):
        return tiles(self.iht, 8, [8*state[0]/(0.6+1.2), 8*state[1]/(0.07+0.07)], [action])

    def get_pi(self, observation):
        t = []
        for i in self.actions:
            actives = self.get_actives(observation, i)
            t.append(np.sum(self.weight[actives]))
        h = np.exp(t)
        h = h/np.sum(h)
        return h   

    def get_gradient_lnpi(self, observation, action, a_actives):
        gradient_lnpi = self.weight[a_actives]
        h = self.get_pi(observation)
        for i in self.actions:
            gradient_lnpi -= h[i] * self.weight[self.get_actives(observation, i)]
        
        return gradient_lnpi

    def choose_action(self, observation):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.choice(self.actions)
        else:
            h = self.get_pi(observation)
            action = np.random.choice(self.actions, p=h)
            return action

    def learn(self, observation, action, delta):
        actives = self.get_actives(observation, action)
        self.weight[actives] += self.alpha * delta * self.get_gradient_lnpi(observation, action, actives)