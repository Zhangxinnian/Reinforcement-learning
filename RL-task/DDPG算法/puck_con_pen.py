import gym
from DDPG算法.puckworld_con_enemy import PuckWorldEnv
from DDPG算法.ddpg_agent import DDPGAgent
from DDPG算法.utils import learning_curve
import numpy as np

env = PuckWorldEnv()
agent = DDPGAgent(env)

agent.load_models(200)
data = agent.learning(max_episode_num=200, display=True)
learning_curve(data, 2, 1,  x_name="episodes", y_name="rewards of episode")
data = agent.learning(max_episode_num = 100, display = True, explore = False)
env.close()