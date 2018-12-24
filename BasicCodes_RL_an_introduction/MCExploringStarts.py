import numpy as np
from tqdm import tqdm

from RLalgorithm import RLalgorithm

Action = ['R', 'L', 'U', 'D']

class MonteCaloES(RLalgorithm):
    def __init__(self, state_size, action_size):
        super(MonteCaloES, self).__init__(state_size, action_size)
        self.Returns = np.zeros(state_size+[action_size])
        self.Counts = np.ones(state_size+[action_size])
        # self.epsilon = 0.05
        self.gamma = 0.95 

    # def choose_action(self, observation):
    #     t = np.random.random()
    #     if t < self.epsilon:
    #         return np.random.choice(self.action_size)
    #     else:
    #         return self.choose_action_best(observation)
    
    def choose_action_best(self, observation):
        x,y = observation
        index = np.random.permutation(self.action_size)
        a = index[np.argmax(self.Q[x,y][index])]
        return a


    def learn(self, episode_state_a, reward):
        count = np.ones(self.state_size+[self.action_size])
        returns = np.zeros(self.state_size+[self.action_size])
        
        #回报衰减，从最后向前衰减
        rewardlist = [self.gamma**i*reward for i in range(len(episode_state_a))][::-1]
        for i,s_a in enumerate(episode_state_a):
            x,y,a = s_a
            
            #判断首次进入该状态
            if count[x,y,a]==1: 
                returns[x,y,a] += rewardlist[i]
            
            count[x,y,a] += 1

        self.Counts += count
        self.Returns += returns

        #更新值函数
        self.Q = self.Returns / self.Counts


    def train(self, max_episode, env):
        for _ in tqdm(range(max_episode)):
            notTerminal = 1
            observation = env.random() #随机初始状态
            count = 0
            episode_state_a = []
            while(notTerminal and count < 100):
                count += 1
                if count > 1:
                    action = self.choose_action_best(observation) #按照贪心策略执行
                else :
                    action = np.random.choice(self.action_size) #随机初始动作

                observation_, reward, notTerminal = env.step(action)
                episode_state_a.append(observation+[action])
                observation = observation_
            self.learn(episode_state_a, reward)
    
    def run(self, env):
        notTerminal = 1
        observation = env.reset()
        count = 0
        while(notTerminal):
            count += 1
            action = self.choose_action_best(observation)
            print(observation, Action[action], count)
            observation_, _ , notTerminal = env.step(action)
            observation = observation_