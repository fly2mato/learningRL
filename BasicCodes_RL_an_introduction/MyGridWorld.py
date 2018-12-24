import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from RLalgorithm import RLalgorithm
from MCExploringStarts import MonteCaloES
from MC_control_epslion import On_policy_MC_control
from SARSA import SARSA
from Qlearning import Qlearning
from DoubleQlearning import DoubleQlearning
from SARSAlambda import SARSAlambda
from Dyna_Q import Dyna_Q


class GridWorld(object):
    def __init__(self, xlen, ylen, goal, start, reward=1):
        self.xlen = xlen
        self.ylen = ylen
        self.goal = goal
        self.reward  = reward
        self.start = start
        self.reset()

    def reset(self):
        self.x, self.y = self.start
        return self.start

    def random(self):
        self.x = np.random.choice(self.xlen)
        self.y = np.random.choice(self.ylen)
        return [self.x, self.y]

    def IsTerminal(self):
        if self.x == self.goal[0] and self.y == self.goal[1]:
            return 1, self.reward
        return 0, -1

    def step(self, action):
        if action == 0 and self.x < self.xlen-1:
            self.x += 1
        if action == 1 and self.x > 0:
            self.x -= 1
        if action == 2 and self.y < self.ylen-1:
            self.y += 1
        if action == 3 and self.y > 0:
            self.y -= 1

        
        IsTerminal, returns = self.IsTerminal()
        return [self.x, self.y], returns, 1-IsTerminal


class WindGridWorld(GridWorld):
    def __init__(self, xlen, ylen, goal, start, reward, windvalue):
        super(WindGridWorld, self).__init__(xlen, ylen, goal, start, reward)
        self.windvalue = windvalue

    def step(self, action):
        self.y += self.windvalue[self.x]
        if action == 0: 
            self.x += 1
        if action == 1:
            self.x -= 1
        if action == 2:
            self.y += 1
        if action == 3:
            self.y -= 1

        self.x = min(max(0, self.x), self.xlen-1)
        self.y = min(max(0, self.y), self.ylen-1)

        IsTerminal = self.IsTerminal()
        reward = (IsTerminal-1) * self.reward
        # reward = IsTerminal * self.reward       
        return [self.x, self.y], reward, 1-IsTerminal

class CliffGridWorld(GridWorld):
    def __init__(self, xlen, ylen, goal, start, goal_reward, cliffPos, cliff_reward):
        super(CliffGridWorld, self).__init__(xlen, ylen, goal, start, goal_reward)
        self.cliff_reward = cliff_reward
        self.safe = np.ones([xlen, ylen])
        for x,y in cliffPos:
            self.safe[x,y] = 0 

    def IsTerminal(self):
        if self.x == self.goal[0] and self.y == self.goal[1]:
            return 1, self.reward
        if self.safe[self.x, self.y] == 0:
            return 1, self.cliff_reward
        return 0, -1

class Maze(GridWorld):
    def __init__(self, xlen, ylen, goal, start, goal_reward, obstaclePos):
        super(Maze, self).__init__(xlen, ylen, goal, start, goal_reward)
        self.go = np.ones([xlen, ylen])
        for x,y in obstaclePos:
            self.go[x,y] = 0 

    def step(self, action):
        x0 = self.x
        y0 = self.y

        if action == 0 and self.x < self.xlen-1:
            self.x += 1
        if action == 1 and self.x > 0:
            self.x -= 1
        if action == 2 and self.y < self.ylen-1:
            self.y += 1
        if action == 3 and self.y > 0:
            self.y -= 1

        if self.go[self.x, self.y] == 0:
            self.x=x0
            self.y=y0

        IsTerminal, returns = self.IsTerminal()
        return [self.x, self.y], returns, 1-IsTerminal
    

if __name__ == "__main__":
    # env = GridWorld(10, 10, [9,6], [1,1], 1)
    # RL = MonteCaloES([10,10], 4)
    # RL = On_policy_MC_control([10,10], 4)
    # env = WindGridWorld(10, 7, [7, 3], [0,3], 1, [0,0,0,1,1,1,2,2,1,0])

    # cliffPos = [[x,0] for x in range(1,9)]
    # env = CliffGridWorld(10, 7, [9,0], [0, 0], 1, cliffPos, -100)


    # # RL = SARSA([10,7], 4, epsilon=0.1, gamma=0.95, alpha=1,expected=True)
    # # RL = Qlearning([10,7], 4)
    # # RL = DoubleQlearning([10,7], 4)
    # # RL = SARSAlambda([10,7], 4, slambda=0.1)
    # RL = Dyna_Q([10,7], 4, sim_n=50)
    # RL.train(500, env)
    # RL.run(env)
    # # RL.plot_episode_time()
    # RL.plot_sumreward_episode()
    
    obstaclePos = [[2,2],[2,3],[2,4],[5,1],[7,3],[7,4],[7,5]]
    env = Maze(9,6,[8,5],[0,3],1,obstaclePos)

    # RL = Qlearning([9,6],4, gamma=0.95)
    # RL = SARSAlambda([9,6],4)
    RL = Dyna_Q([9,6], 4, sim_n=0,gamma=0.95, epsilon=0.1, alpha=0.1, plus=False)
    RL.train(50, env)
    # RL.run(env)
    # RL.plot_episode_time()
    # RL.plot_steps_per_episode()

    RL1 = Dyna_Q([9,6], 4, sim_n=5,gamma=0.95, epsilon=0.1, alpha=0.1, plus=False)
    RL1.train(50, env)

    RL2 = Dyna_Q([9,6], 4, sim_n=50,gamma=0.95, epsilon=0.1, alpha=0.1, plus=False)
    RL2.train(50, env)

    plt.plot(np.arange(1, len(RL.episode_steps)+1), RL.episode_steps)
    plt.plot(np.arange(1, len(RL1.episode_steps)+1), RL1.episode_steps)
    plt.plot(np.arange(1, len(RL2.episode_steps)+1), RL2.episode_steps)
    plt.xlabel('Episodes')
    plt.ylabel('stpes')
    # plt.ylim(0, 500)
    plt.show()