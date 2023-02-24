import torch
import gym
import random
import numpy as np
from DeepQ_Network import ReplayBuffer, DQN
from tqdm import tqdm
import matplotlib.pyplot as plt
import utils


"""test dqn"""
LR = 2e-3
NUM_EPISODES = 500
HIDDEN_DIM = 128
GAMMA = 0.98
EPSILON = 0.01
TARGET_UPDATE = 10
BUFFER_SIZE = 10000
MINIMAL_SIZE = 500
BATCH_SIZE = 64
device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

env_name = "CartPole-v1"
env = gym.make(env_name)
random.seed(0)
np.random.seed(0)
torch.manual_seed(0)
replay_buffer = ReplayBuffer(BUFFER_SIZE)
state_dim = env.observation_space.shape[0]
action_dim = env.action_space.n
agent = DQN(state_dim, HIDDEN_DIM, action_dim, LR, GAMMA, EPSILON, TARGET_UPDATE, device)

return_list = []
for i in range(10):
    with tqdm(total=int(NUM_EPISODES / 10), desc='Iteration %d' % i) as pbar:
        for i_episode in range(int(NUM_EPISODES / 10)):
            episode_return = 0
            state = env.reset()
            done = False
            while not done:
                action = agent.take_action(state)
                next_state, reward, done, _= env.step(action)
                replay_buffer.add(state, action, reward, next_state, done)
                state = next_state
                episode_return += reward
                # 当buffer数据的数量超过一定值后,才进行Q网络训练
                if replay_buffer.size() > MINIMAL_SIZE:
                    b_s, b_a, b_r, b_ns, b_d = replay_buffer.sample(BATCH_SIZE)
                    transition_dict = {
                        'states': b_s,
                        'actions': b_a,
                        'next_states': b_ns,
                        'rewards': b_r,
                        'dones': b_d
                    }
                    agent.update(transition_dict)
            return_list.append(episode_return)
            if(i_episode+1) % 10 == 0:
                pbar.set_postfix({
                    'episode': '%d' % (NUM_EPISODES / 10 * i + i_episode + 1),
                    'return': '%.3f' % np.mean(return_list[-10:])
                })
            pbar.update(1)


episodes_list = list(range(len(return_list)))
plt.plot(episodes_list, return_list)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()

mv_return = utils.moving_average(return_list, 9)
plt.plot(episodes_list, mv_return)
plt.xlabel('Episodes')
plt.ylabel('Returns')
plt.title('DQN on {}'.format(env_name))
plt.show()
