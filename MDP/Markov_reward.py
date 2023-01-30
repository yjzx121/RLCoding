import numpy as np 
np.random.seed(0)

# define state transform matrix
P = [
	[0.9, 0.1, 0.0, 0.0, 0.0, 0.0],
	[0.5, 0.0, 0.5, 0.0, 0.0, 0.0],
	[0.0, 0.0, 0.0, 0.6, 0.0, 0.4],
	[0.0, 0.0, 0.0, 0.0, 0.3, 0.7],
	[0.0, 0.2, 0.3, 0.5, 0.0, 0.0], 
	[0.0, 0.0, 0.0, 0.0, 0.0, 1.0],
]
p = np.array(P)

rewards = [-1, -2, -2, 10, 1, 0]
gamma = 0.5

# giving a list, compute rewards that from start state to end of list
def compute_return(start_index, chain, gamma):
	G = 0.0
	for i in reversed(range(start_index, len(chain))):
		print("现在的i为：%s", i)
		G = gamma * G + rewards[chain[i] - 1]
		print("现在的chain为：%s", chain[i])
		print("现在的G为：%s", G)
		print('* ' * 50)
	return G

# ONE LIST
chain = [1, 2, 3, 6]
start_index = 0

G = compute_return(start_index, chain, gamma)
print("根据本序列计算得到回报为， %s。" % G)