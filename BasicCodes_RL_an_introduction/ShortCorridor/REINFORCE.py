import numpy as np

epsilon = 0.05

class Reinforce(object):
    def __init__(self, actions, alpha=2e-4, gamma=1, baseline=False, alpha_w=2e-2):
        self.alpha = alpha
        self.gamma = gamma
        self.baseline = baseline
        self.alpha_w = alpha_w
        self.w = 0
        self.actions = actions
        self.theta = np.array([-1.0,1.0], dtype=float)
        self.x = np.array([[1,0],
                           [0,1]]) 
        
    def get_pi(self):
        t = self.theta.dot(self.x)
        h = np.exp(t-np.max(t))
        h = h/np.sum(h)
        if min(h) < epsilon:
            imin = np.argmin(h)
            h[imin] = epsilon
            h[1-imin] = 1-epsilon
        return h    
    
    def choose_action(self):
        h = self.get_pi()
        if np.random.uniform() < h[0]:
            return self.actions[0]
        else:
            return self.actions[1]

    def learn(self, states, rewards, actions):
        G = np.zeros(len(rewards))
        G[-1] = rewards[-1]
        for i in range(2, len(rewards)+1):
            G[-i] = self.gamma * G[-i + 1] + rewards[-i]
        
        gamma_pow = 1
        for i in range(len(rewards)):
            h = self.get_pi()
            delta = self.x[:, actions[i]] - self.x.dot(h) # 与这个计算结果一致 (self.x[:,0]*h[0] + self.x[:,1]*h[1])
            if self.baseline:
                delta -= self.w
                self.w += self.alpha_w * gamma_pow * delta           
            self.theta += self.alpha * gamma_pow * G[i] * delta
            gamma_pow *= self.gamma
        

            
