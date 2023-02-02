import time

import gym
from gym import spaces, envs


# env = gym.make('CartPole-v1', render_mode="human")

"""
# Observations
for i_episode in range(20):
    observation = env.reset()
    for t in range(100):
        env.render()
        print(observation)
        action = env.action_space.sample()
        observation, reward, done, truncated, info = env.step(action)
        if done:
            print("Episode finished after {} timesteps".format(t+1))
            break
env.close()


# Space
print(env.action_space)            # >Discrete(2)
print(env.observation_space)       # >Box(4, )
print(env.observation_space.high)  # >[4.8000002e+00 3.4028235e+38 4.1887903e-01 3.4028235e+38]
print(env.observation_space.low)   # >[-4.8000002e+00 -3.4028235e+38 -4.1887903e-01 -3.4028235e+38]

space = spaces.Discrete(8)
x = space.sample()
print(space.contains(x))           # >True
print(space.n == 8)                # >True
"""
from gym.envs.registration import register
register(
    id='CartPole-v2',
    entry_point='gym.envs.classic_control:CartPoleEnv',
    max_episode_steps=200*4,
    reward_threshold=195.0*4,
)
env = gym.make('CartPole-v2', render_mode="human")
env.reset()
env.render()
time.sleep(1)