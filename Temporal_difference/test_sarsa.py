from Sarsa import CliffWalkingEnv, Sarsa
import matplotlib.pyplot as plt
import numpy as np
from tqdm import tqdm


NCOL = 12
NROW = 4
EPSILON = 0.1
ALPHA = 0.1
GAMMA = 0.9

env = CliffWalkingEnv(NCOL, NROW)
np.random.seed(0)
agent = Sarsa(NCOL, NROW, EPSILON, ALPHA, GAMMA)
# 智能体在环境中运行的序列的数量
num_episodes = 500

# 记录每一条序列的回报
return_list = []
for i in range(10):                                                                 # 显示10个进度条
    # tqdm的进度条功能
    with tqdm(total=int(num_episodes / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(num_episodes / 10)):                             # 每个进度条的序列数
            episode_return = 0
            state = env.reset()
            action = agent.take_action(state)
            done = False
            while not done:
                next_state, reward, done = env.step(action)
                next_action = agent.take_action(next_state)
                episode_return += reward                                            # 这里回报的计算不进行折扣因子衰减
                agent.update(state, action, reward, next_state, next_action)
                state = next_state
                action = next_action
            return_list.append(episode_return)

            if (i_episode + 1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (num_episodes / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('Sarsa on {}'.format('Cliff Walking'))
plt.show()
