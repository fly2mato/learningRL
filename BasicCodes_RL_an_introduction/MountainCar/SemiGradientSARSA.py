#值函数逼近SARSA

from TileCodingSoftware import *

import numpy as np

class SemiGradientSARSA(object):
    def __init__(self, ACTIONS, epsilon=0, alpha=0.3, gamma = 1):
        self.actions = ACTIONS
        self.epsilon = epsilon
        self.alpha = alpha/8
        self.gamma = gamma
        self.hash_size = 2048
        self.iht = IHT(self.hash_size)
        self.weight = np.zeros(self.hash_size)

    def get_actives(self, state, action):
        return tiles(self.iht, 8, [8*state[0]/(0.6+1.2), 8*state[1]/(0.07+0.07)], [action])

    def qvalue_function(self, state, action):
        if state[0] >= 0.6:
            return 0
        actives = self.get_actives(state, action)
        return np.sum(self.weight[actives])
    
    def choose_action(self, observation):
        if np.random.binomial(1, self.epsilon) == 1:
            return np.random.choice(self.actions)
        else:
            index = np.random.permutation(len(self.actions))
            qvalues = np.array([self.qvalue_function(observation, a) for a in self.actions])
            a = self.actions[index[np.argmax(qvalues[index])]]
            return a

    def learn(self, observation, action, reward, observation_, action_=None):
        actives = self.get_actives(observation, action)
        delta = self.alpha* (reward + self.gamma * self.qvalue_function(observation_, action_) \
                            - self.qvalue_function(observation, action))
        for a in actives:
            self.weight[a] += delta

    def cost_to_go(self, state):
        costs = []
        for action in self.actions:
            costs.append(self.qvalue_function(state, action))
        return -np.max(costs)

    def print_cost(self, episode, ax):
        grid_size = 40
        positions = np.linspace(-1.2, 0.6, grid_size)
        velocities = np.linspace(-0.07, 0.07, grid_size)
        axis_x = []
        axis_y = []
        axis_z = []
        for position in positions:
            for velocity in velocities:
                axis_x.append(position)
                axis_y.append(velocity)
                axis_z.append(self.cost_to_go((position, velocity)))

        ax.scatter(axis_x, axis_y, axis_z)
        ax.set_xlabel('Position')
        ax.set_ylabel('Velocity')
        ax.set_zlabel('Cost to go')
        ax.set_title('Episode %d' % (episode + 1))


    