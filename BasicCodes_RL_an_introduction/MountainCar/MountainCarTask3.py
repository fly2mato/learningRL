from ActorCritic import Actor,Critic

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

actor = Actor(epsilon=0)
critic = Critic()
Tmax = 1000
for i_episode in range(3000):

    observation = env.reset()
    action = actor.choose_action(observation)

    running_reward = 0
    critic.reset()
    count = 0
    while count < Tmax:
        count += 1
        if RENDER: env.render()

        observation_, reward, done, info = env.step(action)     # reward = -1 in all cases
        
        # print(action, reward, observation_)
        running_reward += reward
        
        if done:
            Tmax = count
            delta = critic.learn(observation, reward, observation_)
            actor.learn(observation, action, delta)
            print(i_episode, running_reward)
            if running_reward > DISPLAY_REWARD_THRESHOLD and i_episode > 1900: RENDER = True     # rendering
            break
        else:
            action_ = actor.choose_action(observation_)
            delta = critic.learn(observation, reward, observation_)
            actor.learn(observation, action, delta)

        observation = observation_
        action = action_

