import numpy as np


class QLearning:
    def __init__(self, ncol, nrow, epsilon, alpha, gamma, n_action=4):
        self.Q_table = np.zeros([nrow * ncol, n_action])                # 初始化Q(s,a)表格
        self.n_action = n_action                                        # 动作个数
        self.alpha = alpha                                              # 学习率
        self.gamma = gamma                                              # 折扣因子
        self.epsilon = epsilon                                          # epsilon-贪婪策略中的参数

    def take_action(self, state):                                       # 选取下一步的操作,具体实现为epsilon-贪婪
        if np.random.random() < self.epsilon:
            action = np.random.randint(self.n_action)
        else:
            action = np.argmax(self.Q_table[state])
        return action

    def best_action(self, state):                                       # 用于打印策略
        Q_max = np.max(self.Q_table[state])
        a = [0 for _ in range(self.n_action)]
        for i in range(self.n_action):                                  # 若两个动作的价值一样,都会记录下来
            if self.Q_table[state, i] == Q_max:
                a[i] = 1
        return a

    def update(self, s0, a0, r, s1):
        # td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]  Sarsa算法更新公式
        td_error = r + self.gamma * self.Q_table[s1].max() - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error