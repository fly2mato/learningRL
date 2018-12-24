from SemiGradientSARSA import SemiGradientSARSA

import gym
import matplotlib.pyplot as plt

DISPLAY_REWARD_THRESHOLD = -90

RENDER = False  # rendering wastes time

env = gym.make('MountainCar-v0')
env.seed(1)     # reproducible, general Policy gradient has high variance
env = env.unwrapped

print(env.action_space)
print(env.observation_space)
print(env.observation_space.high)
print(env.observation_space.low)

rl = SemiGradientSARSA([0,1,2])

for i_episode in range(2000):

    observation = env.reset()
    action = rl.choose_action(observation)

    running_reward = 0

    while True:
        if RENDER: env.render()

        observation_, reward, done, info = env.step(action)     # reward = -1 in all cases
        
        running_reward += reward
        
        if done:
            rl.learn(observation, action, reward, observation_)
            print(i_episode, running_reward)
            if running_reward > DISPLAY_REWARD_THRESHOLD and i_episode > 1900: RENDER = True     # rendering
            break
        else:
            action_ = rl.choose_action(observation_)
            rl.learn(observation, action, reward, observation_, action_)

        observation = observation_
        action = action_
