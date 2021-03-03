from random import random,choice
import gym
from gym import Env
import numpy as np
from typing import List
from collections import namedtuple
import random
from tqdm import tqdm

class State(object):
    def __init__(self, name):
        self.name = name

class Transition(object):
    def __init__(self, s0, a0, reward:float, is_done:bool, s1):
        self.data = [s0, a0, reward, is_done, s1]

    def __iter__(self):
        return iter(self.data)

    def __str__(self):
        return 's:{0:<3} a:{1:<3} r:{2:<4} is_end:{3:<5} s1:{4:<3}'.\
            format(self.data[0], self.data[1], self.data[2], self.data[3], self.data[4])

    @property
    def s0(self): return self.data[0]

    @property
    def a0(self): return self.data[1]

    @property
    def reward(self): return self.data[2]

    @property
    def is_done(self): return  self.data[3]

    @property
    def s1(self): return  self.data[4]

class Episode(object):
    def __init__(self, e_id:int = 0) -> None:
        self.total_reward = 0 #Total reward
        self.trans_list = [] #State transition list
        self.name = str(e_id) #You can give Episode a name: "Successfully break through, but sadly fail?

    def push(self, trans:Transition) -> float:
        '''
        Send a state transition into the state sequence and return the current total reward value of the sequence
        '''
        self.trans_list.append(trans)
        self.total_reward += trans.reward #Total reward without decay
        return self.total_reward

    @property
    def len(self):
        return len(self.trans_list)

    def __str__(self):
        return "episode {0:<4} {1:>4} steps,total reward:{2:<8.2f}". \
            format(self.name, self.len, self.total_reward)

    def print_detail(self):
        print('detail of ({0}):'.format(self))
        for i, trans in enumerate(self.trans_list):
            print('step{0:<4}'.format(i), end=' ')
            print(trans)

    def pop(self) -> Transition:
        '''
        Normally this method shouldn't be invoked
        '''
        if self.len > 1:
            trans = self.trans_list.pop()
            self.total_reward -= trans.reward
            return trans
        else:
            return None

    def is_complete(self) -> bool:
        '''
        Check if an episode is an complete episode
        '''
        if self.len == 0:
            return False
        return self.trans_list[self.len-1].is_done

    def sample(self, batch_size = 1):
        '''
        Randomly generate one trans
        '''
        return  random.sample(self.trans_list, k = batch_size)

    def __len__(self) -> int:
        return self.len

class Experience(object):
    '''
    this class is used to record the whole experience of an agent organized
    by an episode list. agent can randomly sample transitions or episodes from
    its experience
    '''
    def __init__(self, capacity:int = 20000):
        self.capacity = capacity  #Capacity: refers to the total number of trans
        self.episodes = [] #Episode list
        self.next_id = 0 #Id of the next episode
        self.total_trans = 0 #Total number of state transitions

    def __str__(self):
        return 'exp info:{0:5} episodes, memory usage {1}/{2}'.\
            format(self.len, self.total_trans, self.capacity)

    def __len__(self):
        return self.len

    @property
    def len(self):
        return len(self.episodes)

    def _remove(self, index = 0):
        '''
        emove an episode, defautly the first one.
           args:
               the index of the episode to remove
           return:
               if exists return the episode else return None
        '''
        if index > self.len -1:
            raise(Experience('invalid index'))
        if self.len > 0:
            episode = self.episodes[index]
            self.episodes.remove(episode)
            self.total_trans -= episode.len
            return episode
        else:
            return None

    def _remove_first(self):
        self._remove(index=0)

    def push(self, trans):
        '''
        Push a state transition
        '''
        if self.capacity <= 0:
            return
        while self.total_trans >= self.capacity: #Could there be a free episode?
            episode = self._remove_first()
        cur_episode = None
        if self.len == 0 or self.episodes[self.len-1].is_complete():
            cur_episode = Episode(self.next_id)
            self.next_id += 1
            self.episodes.append(cur_episode)
        else:
            cur_episode = self.episodes[self.len-1]
        self.total_trans += 1
        return cur_episode.push(trans) #return total reward of an episode

    def sample(self, batch_size = 1): #sample transition
        '''
        Randomly sample some transitions from agent's experience.abs
        Args:
            number of transitions need to be sampled
        Return:
            list of Transition
        '''
        sample_trans = []
        for _ in range(batch_size):
            index = int(random.random()*self.len)
            sample_trans += self.episodes[index].sample()
        return sample_trans

    def sample_episode(self, episode_num = 1): # sample episode
        '''
        Randomly obtain a certain number of complete Episodes
        '''
        return random.sample(self.episodes, k= episode_num)

    @property
    def last_episode(self):
        if self.len > 0:
            return self.episodes[self.len-1]
        return None

class Agent(object):
    '''
    Base class of agent
    '''
    def __init__(self, env: Env = None, capacity = 10000):
        #Save some environmental information that the Agent can observe and the lessons learned
        self.env = env #Establish references to environment objects
        self.obs_sapce = env.observation_space if env is not None else None
        self.action_sapce = env.action_space if env is not None else None
        self.S = [i for i in range(self.obs_sapce.n)]
        self.A = [i for i in range(self.action_sapce.n)]
        self.experience = Experience(capacity = capacity)
        #It is relatively convenient to have a variable to record the current state of the agent.
        # Pay attention to the maintenance and update of this variable
        self.sate = None #Individual's current state

    def policy(self, A, s= None, Q = None, epsilon = None):
        '''
        Uniform random strategy
        '''
        return random.sample(self.A, k=1)[0]

    def perform_policy(self, s, Q=None, epsion = 0.05):
        action = self.policy(self.A, s, Q, epsion)
        return int(action)

    def act(self, a0):
        s0 = self.sate
        s1, r1, is_done, info = self.env.step(a0)
        # Todo add extra code here
        trans = Transition(s0, a0, r1, is_done, s1)
        total_reward = self.experience.push(trans)
        self.sate = s1
        return s1, r1, is_done, info, total_reward

    def learning_method(self, lambda_=0.9, gamma=0.9, alpha=0.5, epsilon=0.2, display=False):
        '''
        This is a learning method without learning ability
        Specifically for the learning method of an algorithm,
        the return value needs to be a two-dimensional tuple: (time step of a state sequence, total reward of the state sequence)
        '''
        self.state = self.env.reset()
        s0 = self.state
        if display:
            self.env.render()
        a0 = self.perform_policy(s0, epsilon)
        time_in_episode, total_reward = 0, 0
        is_done = False
        while not is_done:
            # add code here
            s1, r1, is_done, info, total_reward = self.act(a0)
            if display:
                self.env.render()
            a1 = self.perform_policy(s1, epsilon)
            # add your extra code here
            s0, a0 = s1, a1
            time_in_episode += 1
        if display:
            print(self.experience.last_episode)
        return time_in_episode, total_reward

    def learning(self, lambda_=0.9, epsilon=None, decaying_epsilon=True, gamma=0.9,
                 alpha=0.1, max_episode_num=800, display=False):
        total_time, episode_reward, num_episode = 0, 0, 0
        total_times, episode_rewards, num_episodes = [], [], []
        for i in tqdm(range(max_episode_num)):
            if epsilon is None:
                epsilon = 1e-10
            elif decaying_epsilon:
                epsilon = 1.0 / (1 + num_episode)
            time_in_episode, episode_reward = self.learning_method(lambda_=lambda_, \
                                                                   gamma=gamma, alpha=alpha, epsilon=epsilon,
                                                                   display=display)
            total_time += time_in_episode
            num_episode += 1
            total_times.append(total_time)
            episode_rewards.append(episode_reward)
            num_episodes.append(num_episode)
        # self.experience.last_episode.print_detail()
        return total_times, episode_rewards, num_episodes

    def sample(self, batch_size = 64):
        '''
        Random sampling
        '''
        return self.experience.sample(batch_size)

    @property
    def total_trans(self):
        '''
        Get the total number of state transitions recorded in the Experience
        '''
        return self.experience.total_trans

    def last_episode_detail(self):
        self.experience.last_episode.print_detail()
        


