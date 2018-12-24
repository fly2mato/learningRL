import numpy as np
from tqdm import tqdm
import matplotlib.pyplot as plt

from env_short_corrider import ShortCorrider
from REINFORCE import Reinforce

env = ShortCorrider()
# rl = Reinforce(env.actions)
rl = Reinforce(env.actions, baseline=True)

episodes = 1000
trial = 30
rewards_sum = np.zeros(1000)
for i in tqdm(range(trial)):
    for ep in tqdm(range(episodes)):
        if ep == 50:
            pass
        env.reset()
        states = []
        actions = []
        rewards = []
        while True:
            action = rl.choose_action()
            reward, terminal = env.step(action)

            states.append(env.state)
            actions.append(action)
            rewards.append(reward)

            if terminal:
                rl.learn(states, rewards, actions)
                break
        rewards_sum[ep] += np.sum(rewards)

plt.figure()
plt.plot(np.arange(episodes)+1,rewards_sum/trial)
plt.show()

print(rl.get_pi())
print(rl.w)