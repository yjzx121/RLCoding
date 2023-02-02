import numpy as np
import gym
import time


def get_action(weights, observation):                       # 根据权值对当前状态做出决策
    print("weights shape: ", weights.shape)
    print("observ: ", observation.shape)
    wxb = np.dot(weights[:4], observation) + weights[4]     # 计算加权和
    if wxb >= 0:                                            # 加权和大于0时选取动作1，否则选取0
        return 1
    else:
        return 0


def get_sum_reward_by_weights(env, weights):
    # 测试不同权值的控制模型有效控制的持续时间（或奖励）
    observation = env.reset()                                # 重置初始状态
    sum_reward = 0                                           # 记录总的奖励
    for t in range(1000):
        # time.sleep(0.01)
        # env.render()
        action = get_action(weights, observation)                   # 获取当前权值下的决策动作
        observation, reward, done, info = env.step(action)          # 执行动作并获取这一动作下的下一时间步长状态
        print("observation: ", observation)
        print("reward: ", reward)
        print("done: ", done)
        print("info: ", info)
        sum_reward += reward
        # print(sum_reward, action, observation, reward, done, info)
        if done:        # 如若游戏结束，返回
            break
    return sum_reward


def get_weights_by_random_guess():
    # 选取随机猜测的5个随机权值
    return np.random.rand(5)


def get_weights_by_hill_climbing(best_weights):
    # 通过爬山算法选取权值（在当前最好权值上加入随机值）
    return best_weights + np.random.normal(0, 0.1, 5)


def get_best_result(algo="random_guess"):
    env = gym.make("CartPole-v1")
    np.random.seed(10)
    best_reward = 0 # 初始最佳奖励
    best_weights = np.random.rand(5)    # 初始权值为随机取值

    for iter in range(10000):   # 迭代10000次
        cur_weights = None

        if algo == "hill_climbing":     # 选取动作决策的算法
            # print(best_weights)
            cur_weights = get_weights_by_hill_climbing(best_weights)
        else:               # 若为随机猜测算法，则选取随机权值
            cur_weights = get_weights_by_random_guess()
        # 获取当前权值的模型控制的奖励和
        cur_sum_reward = get_sum_reward_by_weights(env, cur_weights)

        # print(cur_sum_reward, cur_weights)
        # 更新当前最优权值
        if cur_sum_reward > best_reward:
            best_reward = cur_sum_reward
            best_weights = cur_weights
        # 达到最佳奖励阈值后结束
        if best_reward >= 200:
            break

    print(iter, best_reward, best_weights)
    return best_reward, best_weights


# 程序从这里开始执行
print(get_best_result("hill_climbing")) # 调用爬山算法寻优并输出结果
