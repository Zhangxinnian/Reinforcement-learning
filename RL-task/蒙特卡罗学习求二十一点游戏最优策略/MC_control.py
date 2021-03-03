import os
import sys
path = os.path.dirname(__file__)
sys.path.append(path)
from blackjack import Player, Dealer, Arena
from utils import  str_key, set_dict, get_dict
from utils import  draw_value, draw_policy
from utils import  epsilon_greedy_policy
import math

class MC_Player(Player):
    '''
    Players with Monte Carlo control capabilities
    '''
    def __init__(self, name = '', A = None, display = False):
        super(MC_Player, self).__init__(name, A, display)
        self.Q = {} #The value of a certain state behavior pair, used when the strategy is iterated
        self.Nsa = {} #Nsa count: the number of occurrences of a certain transition behavior pair
        self.total_learning_times = 0
        self.policy = self.epsilon_greedy_policy
        self.learning_method = self.learn_Q #Have my own learning method

    def learn_Q(self, episode, r): #Learn the Q value from the state sequence
        '''
        Learn from an episode
        '''
        for s, a in episode:
            nsa = get_dict(self.Nsa, s, a)
            set_dict(self.Nsa, nsa+1, s, a)
            q = get_dict(self.Q, s, a)
            set_dict(self.Q, q+(r-q)/(nsa+1), s, a)
        self.total_learning_times += 1

    def reset_memory(self):
        '''
        Forget past learning experience
        '''
        self.Q.clear()
        self.Nsa.clear()
        self.total_learning_times = 0

    def epsilon_greedy_policy(self, dealer, epsilon = None):
        '''
        The greedy strategy here is with epsilon parameters
        '''
        player_points, _ = self.get_points()
        if player_points >= 21:
            return self.A[1]
        if player_points < 12:
            return self.A[0]
        else:
            A, Q = self.A, self.Q
            s = self.get_state_name(dealer)
            if epsilon is None:
                epsilon = 1.0 / (1 + 4 * math.log10(1 + player.total_learning_times))
            return epsilon_greedy_policy(A, s, Q, epsilon)


A=["继续叫牌","停止叫牌"]
display = False
player = MC_Player(A = A, display = display)
dealer = Dealer(A = A, display = display)
arena = Arena(A = A, display = display)
arena.play_games(dealer = dealer, player = player,num = 200000, show_statistic = True)

draw_value(player.Q, useable_ace = True, is_q_dict = True, A = player.A)
draw_policy(epsilon_greedy_policy, player.A, player.Q, epsilon = 1e-10, useable_ace = True)
draw_value(player.Q, useable_ace = False, is_q_dict = True, A = player.A)
draw_policy(epsilon_greedy_policy, player.A, player.Q, epsilon = 1e-10, useable_ace = False)


