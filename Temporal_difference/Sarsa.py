import numpy as np


class CliffWalkingEnv:
    def __init__(self, ncol, nrow):
        self.ncol = ncol
        self.nrow = nrow
        # 记录当前智能体位置的横坐标
        self.x = 0
        # 记录当前智能体位置的纵坐标
        self.y = self.nrow - 1

    # 外部调用这个函数来改变当前位置
    def step(self, action):
        # 4种动作, change[0]:上, change[1]:下, change[2]:左, change[3]:右。坐标系原点(0,0)
        # 定义在左上角
        change = [[0, -1], [0, 1], [-1, 0], [1, 0]]
        self.x = min(self.ncol - 1, max(0, self.x + change[action][0]))
        self.y = min(self.nrow - 1, max(0, self.y + change[action][1]))

        next_state = self.y * self.ncol + self.x
        reward = -1
        done = False

        if self.y == self.nrow - 1 and self.x > 0:                 # 下一个位置在悬崖或者目标
            done = True
            if self.x != self.ncol - 1:
                reward = -100
        return next_state, reward, done

    def reset(self):                                               # 回归初始状态,坐标轴原点在左上角
        self.x = 0
        self.y = self.nrow - 1
        return self.y * self.ncol + self.x


"""
然后我们来实现 Sarsa 算法，主要维护一个表格Q_table()，用来储存当前策略下所有状态动作对的价值，
在用 Sarsa 算法和环境交互时，用-贪婪策略进行采样，在更新 Sarsa 算法时，使用时序差分的公式。
我们默认终止状态时所有动作的价值都是 0，这些价值在初始化为 0 后就不会进行更新。
"""


class Sarsa:
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

    def update(self, s0, a0, r, s1, a1):
        td_error = r + self.gamma * self.Q_table[s1, a1] - self.Q_table[s0, a0]
        self.Q_table[s0, a0] += self.alpha * td_error


