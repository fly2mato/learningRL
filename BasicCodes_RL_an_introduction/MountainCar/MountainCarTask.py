import numpy as np
from tqdm import tqdm


from MountainCar import MountainCar
from SemiGradientSARSA import SemiGradientSARSA

# import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d.axes3d import Axes3D
from math import floor

env = MountainCar()
rl = SemiGradientSARSA(env.actions)

Kmax = float('inf')

episodes = 1000
plot_episodes = [0, 99, episodes - 1]
fig = plt.figure(1, figsize=(40, 10))
axes = [fig.add_subplot(1, len(plot_episodes), i+1, projection='3d') for i in range(len(plot_episodes))]

position = []
velocity = []
actions = []

for ep in tqdm(range(episodes)):
    steps = 0
    env.reset()
    observation = env.state
    action = rl.choose_action(observation)
    while steps <= Kmax :
        steps += 1
        observation_, reward = env.step(action)
        
        if env.isTerminal():
            rl.learn(observation, action, reward, observation_)
            break
        else:
            action_ = rl.choose_action(observation_)
            rl.learn(observation, action, reward, observation_, action_)
        
        observation = observation_
        action = action_

        if ep == episodes - 1:
            position.append(observation[0])
            velocity.append(observation[1])
            actions.append(action)
    
    Kmax = steps
    if ep in plot_episodes:
        rl.print_cost(ep, axes[plot_episodes.index(ep)])

# figure(3, figsize=(10,10))
# plt.scatter(position, velocity)
zz = []
for i in range(len(position)):
    zz.append(rl.cost_to_go((position[i], velocity[i])))
axes[-1].scatter(position, velocity, zz, c='r', linewidth = 3)

fig = plt.figure(2, figsize=(40, 10))
plt.subplot(131)
plt.plot(np.arange(0, len(position)), position)
plt.subplot(132)
plt.plot(np.arange(0, len(velocity)), velocity)
plt.subplot(133)
plt.plot(np.arange(0, len(actions)), actions)

plt.show()



