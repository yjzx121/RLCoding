from test_sarsa import Sarsa_agent, env
from test_muti_sarsa import MutiSarsa_agent
from test_qlearning import QLearning_agent


def print_agent(agent, env, action_meaning, disaster=[], end=[]):
    for i in range(env.nrow):
        for j in range(env.ncol):
            if (i * env.ncol + j) in disaster:
                print('****', end=' ')
            elif (i * env.ncol + j) in end:
                print('EEEE', end=' ')
            else:
                a = agent.best_action(i * env.ncol + j)
                pi_str = ''
                for k in range(len(action_meaning)):
                    pi_str += action_meaning[k] if a[k] > 0 else 'o'
                print(pi_str, end=' ')
        print()


action_meaning = ['^', 'v', '<', '>']

for agt in [Sarsa_agent, MutiSarsa_agent, QLearning_agent]:
    print('Sarsa算法最终收敛得到的策略为：')
    print_agent(Sarsa_agent, env, action_meaning, list(range(37, 47)), [47])