import gym
from PuckWorld游戏.puckworld import PuckWorldEnv
from PuckWorld游戏.agents import DQNAgent
from PuckWorld游戏.utils import learning_curve

env = PuckWorldEnv()
agent = DQNAgent(env)

data = agent.learning(gamma=0.99,epsilon=1,decaying_epsilon=True, alpha=1e-3,max_episode_num=1000, display=False)
learning_curve(data, 2, 1,title="DQNAgent performance on PuckWorld",x_name="episodes",y_name="rewards of episode")

data = agent.learning(gamma=0.99, epsilon=0.0001, decaying_epsilon=False, alpha=0.001, max_episode_num=20,display=True)
env.close()