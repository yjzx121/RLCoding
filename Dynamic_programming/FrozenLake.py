import time
from Cliff_Walking import print_P
import gym
from policy_iteration import PolicyIteration, print_agent


# 创建环境
env = gym.make("FrozenLake-v1", render_mode="human")
env = env.unwrapped                        # 解封装才能访问状态转移矩阵p
env.reset()
env.render()                               # 环境渲染，通常是弹窗显示或打印出可视化的环境


holes = set()
ends = set()

print_P(env.P, 4, 4)

for s in env.P:
    for a in env.P[s]:
        for s_ in env.P[s][a]:
            # 获得奖励+1，代表是目标
            if s_[2] == 1.0:
                ends.add(s_[1])
            if s_[3] == True:
                holes.add(s_[1])

holes = holes - ends
print("冰洞的索引:", holes)
print("目标的索引:", ends)

# 查看目标左边一格的状态转移信息
for a in env.P[14]:
    print(env.P[14][a])


action_meaning = ['<', 'v', '>', '^']
theta = 1e-5
gamma = 0.9
agent = PolicyIteration(env, theta, gamma)
agent.policy_iteration()
print_agent(agent, action_meaning, [5, 7, 11, 12], [15])
time.sleep(1000)

