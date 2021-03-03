import os
import sys
path = os.path.dirname(__file__)
sys.path.append(path)
import gym
from gym import Env
from gridworld import WindyGridWorld
from core import Agent

env = WindyGridWorld() #Generate stylized sub-world environment objects
env.reset() #Reset environment objects
env.render() #Display the visual interface of environment objects

agent = Agent(env, capacity = 10000)
data = agent.learning(max_episode_num= 180, display= False)