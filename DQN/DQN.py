import tensorflow as tf
import gym
import numpy as np
from tqdm import tqdm
import matplotlib
import matplotlib.pyplot as plt
import os

#=============Auxiliary Functions ================
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
#=====================================================================================


def networkInit(sess,firstFun):
    saver = tf.train.Saver()
    if not os.path.isdir('./paralog/'):
        os.mkdir('./paralog/')

    try:
        if not firstFun:
            saver.restore(sess, './paralog/model.ckpt')
            print('restore from disk!')
        else:
            sess.run(tf.global_variables_initializer())
    except:
        sess.run(tf.global_variables_initializer())

    return saver


def dqn(envname='CartPole-v0', lr=0.1, firstRun=False, 
        steps_per_epoch=6000, epochs=50, gamma=0.99, max_ep_len=2000,
        train_v_iters=20, epsilon=0.1):

    env = gym.make(envname)

    observation_dim = env.observation_space.shape[0]
    action_dim = env.action_space.n

    #build network
    x_ph = placeholder_from_space(env.observation_space)
    target_ph = placeholder(action_dim)
    qvalues = mlp(x_ph, [64,action_dim])
    action = tf.argmax(qvalues, axis=1)
    qvalue_max = tf.reduce_max(qvalues, axis=1)

    #loss function
    qloss = tf.reduce_max(tf.squared_difference(target_ph, qvalues))
    train_loss = tf.train.AdamOptimizer(learning_rate=lr).minimize(qloss)

    sess = tf.Session()
    saver = networkInit(sess, firstRun)

    def rl_update():
        s = data[:,:observation_dim]
        s_ = data[:,-observation_dim:]
        a = data[:,observation_dim].astype(int)
        r = data[:,observation_dim+1]

        target = sess.run(qvalues, feed_dict={x_ph:s})
        q_next = sess.run(qvalue_max, feed_dict={x_ph:s_})
        target[np.arange(steps_per_epoch), a] = np.array(r) + gamma * q_next
        for i in range(train_v_iters):
            _, loss = sess.run([train_loss, qloss], feed_dict={x_ph:s, target_ph:target})
        return loss

    def rl_update_replay():
        index = np.random.choice(steps_per_epoch, size=np.int(steps_per_epoch/10), replace=False)
        s = data[index,:observation_dim]
        s_ = data[index,-observation_dim:]
        a = data[index,observation_dim].astype(int)
        r = data[index,observation_dim+1]

        target = sess.run(qvalues, feed_dict={x_ph:s})
        q_next = sess.run(qvalue_max, feed_dict={x_ph:s_})
        target[np.arange(len(index)), a] = np.array(r) + gamma * q_next
        for i in range(train_v_iters):
            _, loss = sess.run([train_loss, qloss], feed_dict={x_ph:s, target_ph:target})
        return loss


    episode_reward_loss = []
    RENDER = False
    data = np.zeros([steps_per_epoch, 2*observation_dim + 2])
    for epoch in tqdm(range(epochs)):
        count = 0
        sum_rew = 0
        mean_rew = 0
        iternum = 1

        observation = env.reset()
        for index in tqdm(range(steps_per_epoch)):
            if RENDER: 
                #env.render()
                pass
                                
            if (np.random.binomial(1, epsilon) == 0):
                a = sess.run(action, feed_dict={x_ph:observation.reshape(1,-1)})[0]
            else:
                a = env.action_space.sample()

            observation_, reward, done, _ = env.step(a)
            data[index,:] = np.hstack((observation, a, reward, observation_))

            observation = observation_
            sum_rew += reward
            count += 1
            if (count > max_ep_len or done):
                if sum_rew > 180: RENDER=True
                count = 0
                mean_rew += (sum_rew - mean_rew)/iternum
                iternum += 1
                sum_rew = 0
                observation = env.reset()
            
            if epoch > 1:
                rl_update_replay()
        
        print(mean_rew, iternum)    
        loss = rl_update()
        episode_reward_loss.append([mean_rew, loss, iternum])

    saver.save(sess, "./paralog/model.ckpt")
    print(episode_reward_loss)

    epis = np.array(episode_reward_loss)
    plt.figure()
    plt.subplot(211)
    plt.title('reward per episode')
    plt.plot(np.arange(epis.shape[0]), epis[:, 0])
    plt.subplot(212)
    plt.title('loss')
    plt.plot(np.arange(epis.shape[0]), epis[:, 1])
    plt.show()

dqn(firstRun=True, epochs=100, steps_per_epoch=8000,train_v_iters=1, lr=0.01)
# dqn(firstRun=True, epochs=50, steps_per_epoch=5000,train_v_iters=80, lr=0.01)
# dqn(firstRun=False, epochs=10, steps_per_epoch=10)


