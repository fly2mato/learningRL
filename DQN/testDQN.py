# coding: utf-8

# In[1]:


import tensorflow as tf
import gym
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt

# In[2]:


env = gym.make('CartPole-v0')
# print(env.action_space)
# print(env.observation_space)

# In[3]:


from gym.spaces import Discrete, Box

def combined_shape(length, shape=None):
    if shape is None:
        return (length,)
    return (length, shape) if np.isscalar(shape) else (length, *shape)

def placeholder(dim=None):
    return tf.placeholder(dtype=tf.float32, shape=combined_shape(None,dim))

def placeholders(*args):
    return [placeholder(dim) for dim in args]

def placeholder_from_space(space):
    if isinstance(space, Box):
        return placeholder(space.shape)
    elif isinstance(space, Discrete):
        return tf.placeholder(dtype=tf.int32, shape=(None,))
    raise NotImplementedError

def placeholders_from_spaces(*args):
    return [placeholder_from_space(space) for space in args]

def mlp(x, hidden_layers=(32,), activation=tf.tanh, output_activation=None):
    for h in hidden_layers[:-1]:
        x = tf.layers.dense(x, units=h, activation=activation)
    return tf.layers.dense(x, units=hidden_layers[-1], activation=output_activation)


# In[4]:


x_ph = placeholder_from_space(env.observation_space)

action_dim = env.action_space.n
target_ph = placeholder(action_dim)
qvalues = mlp(x_ph, [64,action_dim])
action = tf.argmax(qvalues, axis=1)
qvalue_max = tf.reduce_max(qvalues, axis=1)


qloss = tf.reduce_max(tf.squared_difference(target_ph, qvalues))

qf_lr = 0.01
train_loss = tf.train.AdamOptimizer(learning_rate=qf_lr).minimize(qloss)

sess = tf.Session()

import os
saver = tf.train.Saver()
if os.path.isdir('./paralog/'):
    pass
else:
    os.mkdir('./paralog/')

try:
    saver.restore(sess, './paralog/model.ckpt')
    print('restore!')
except:
    sess.run(tf.global_variables_initializer())


# In[5]:


def update():
    x = np.array(obs)
    target = sess.run(qvalues, feed_dict={x_ph:x[:-1]})
    q_next = sess.run(qvalue_max, feed_dict={x_ph:x[1:]})
    target[np.arange(target.shape[0]), act] = np.array(rew) + gamma * q_next
    _, loss = sess.run([train_loss, qloss], feed_dict={x_ph:x[:-1], target_ph:target})
    return loss


# In[6]:






max_iter = 1500
epsilon = 0.1
gamma = 0.95
episode = 2000
episode_reward_loss = []

RENDER = False
for i in tqdm(range(episode)):
    count = 0
    sum_rew = 0
    obs = []
    act = []
    rew = []

    observation = env.reset()
    obs.append(observation)
    while True:
        if RENDER: 
            env.render()
            pass
                            
        if (np.random.binomial(1, epsilon) == 0):
            a = sess.run(action, feed_dict={x_ph:observation.reshape(1,-1)})[0]
        else:
            a = env.action_space.sample()
            # = np.random.choice(np.arange(action_dim))
        observation, reward, done, _ = env.step(a)
        obs.append(observation)
        act.append(a)
        rew.append(reward)

        sum_rew += reward
        count += 1
        if (count > max_iter or done):
            loss = update()
            episode_reward_loss.append([sum_rew, loss])
            if sum_rew > 180: RENDER=True
            #print([i,sum_rew, loss])
            break

saver.save(sess, "./paralog/model.ckpt")
print(episode_reward_loss)