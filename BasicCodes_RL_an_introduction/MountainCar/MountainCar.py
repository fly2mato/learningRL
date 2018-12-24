import numpy as np
from tqdm import tqdm

class MountainCar(object):
    def __init__(self):
        self.actions = np.array([1,-1,0], dtype=int)
        self.boundx = np.array([-1.2, 0.5])
        self.boundv = np.array([-0.07, 0.07])
        self.goalx = 0.5
        self.reset()
    
    def reset(self):
        self.state = (np.random.uniform(-0.6, -0.4), 0)

    def step(self, action):
        v = self.state[1] + 0.001 * action - 0.0025 * np.cos(3*self.state[0])
        v = max(min(v, self.boundv[1]), self.boundv[0])
        x = self.state[0] + v
        x = min(max(x, self.boundx[0]), self.boundx[1])
        reward = -1

        if self.state[0] == self.goalx:
            v = 0

        self.state = (x, v)
        return self.state, reward

    def isTerminal(self):
        if self.state[0] == self.goalx:
            return True
        else:
            return False
    
    

