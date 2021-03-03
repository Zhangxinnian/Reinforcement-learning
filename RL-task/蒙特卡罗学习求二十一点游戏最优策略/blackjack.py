from random import shuffle
from queue import Queue
from tqdm import tqdm
import math
import matplotlib.pyplot as plt
import numpy as np
from mpl_toolkits.mplot3d import Axes3D
import os
import sys
path = os.path.dirname(__file__)
sys.path.append(path)
from utils import str_key, set_dict, get_dict

class Gamer():
    '''
    Gamer
    '''
    def __init__(self,name = '', A = None, display = False):
        self.name = name
        self.cards = [] #Cards in hand
        self.display = display #Whether to display game text information
        self.policy = None # Strategy
        self.learning_method = None #Learning method
        self.A = A #Action space

    def __str__(self):
        return self.name

    def _value_of(self, card):
        '''
        Judge the numerical value of the card according to the characters of the card,
        A is output as 1, JQK is 10,
        and the rest are taken according to the numerical value corresponding to the characters of the card
        Args:
            card: Card information str
        Return:
            Card size value int, A reutrn 1
        '''
        try:
            v = int(card)
        except:
            if card == 'A':
                v = 1
            elif card in ['J','Q','K']:
                v = 10
            else:
                v = 0
        finally:
            return v

    def get_points(self):
        '''
        Count the score of a hand, if 1 point of A is used, return True at the same time
        Args:
            cards The CARDS in the hands of the declarer or player list ['A','10','3']
        Return:
            tuple (The total number of returned cards, whether reusable Ace is used)
            For example ['A','10','3'] return (14,False)
                        ['A','10'] return (21,True)
        '''
        num_of_useable_ace = 0 #Not got Ace by default
        total_point = 0 #Total value
        cards = self.cards
        if cards is None:
            return 0, False
        for card in cards:
            v = self._value_of(card)
            if v == 1:
                num_of_useable_ace += 1
                v = 11
            total_point += v
        while total_point > 21 and num_of_useable_ace > 0:
            total_point -= 10
            num_of_useable_ace -= 1
        return total_point,bool(num_of_useable_ace)

    def receive(self,cards = []): #Player gets one or more cards
        cards = list(cards)
        for card in cards:
            self.cards.append(card)

    def discharge_cards(self): #The player empties the cards in his hand and throws the cards
        '''
        Throws the cards
        '''
        self.cards.clear()

    def cards_info(self): #Information about the hand of the player
        '''
        Display the cards information
        '''
        self._info('{}{}现在的牌:{}\n'.format(self.role, self, self.cards))

    def _info(self, msg):
        if self.display:
            print(msg, end='')


class Dealer(Gamer):
    '''
    Dealer
    '''
    def __init__(self,name = '', A = None, display = False):
        super(Dealer,self).__init__(name, A, display)
        self.role = '庄家' #Roles
        self.policy = self.dealer_policy #Dealer's tactics

    def first_card_value(self): #Display the first  bright card
        if self.cards is None or len(self.cards) == 0:
            return 0
        return self._value_of(self.cards[0])

    def dealer_policy(self, Dealer = None): #Details of the dealer's tactics
        action = ''
        dealer_points, _ = self.get_points()
        if dealer_points >= 17:
            action = self.A[1]  # 'Stop bidding'
        else:
            action = self.A[0] # 'Keep bidding'
        return action

class Player(Gamer):
    '''
    Player
    '''
    def __init__(self, name = '', A = None, display = False):
        super(Player, self).__init__(name, A, display)
        self.policy = self.naive_policy
        self.role = '玩家'

    def get_state(self,dealer):
        dealer_first_card_value = dealer.first_card_value()
        player_points, useable_ace = self.get_points()
        return dealer_first_card_value,player_points,useable_ace

    def get_state_name(self,dealer):
        return str_key(self.get_state(dealer))

    def naive_policy(self,dealer = None):
        player_points, _ = self.get_points()
        if player_points < 20:
            action = self.A[0]
        else:
            action = self.A[1]
        return action


class Arena():
    '''
    Responsible for game management
    '''
    def __init__(self,display = None, A = None):
        self.cards = ['A','2','3','4','5','6','7','8','9','10','J','Q','K']*4
        self.card_q = Queue(maxsize = 52 ) # Shuffled cards
        self.cards_in_pool = [] #Open cards that have been used
        self.display = display
        self.episodes = [] #List of match information generated
        self.load_cards(self.cards) #Load 52 cards in the initial state into the dealer
        self.A = A #Get action space

    def load_cards(self, cards):
        '''
        Shuffle the collected cards and reload them into the dealer
        Args:
            cards Multiple cards to be loaded into the dealer list
        Return:
            None
        '''
        shuffle(cards) #Shuffle cards
        for card in cards: #The deque data structure can only be added one by one
            self.card_q.put(card)
        cards.clear() #The original card is empty
        return

    def reward_of(self, dealer, player):
        '''
        Judge the player's reward value, with player and dealer's card point information
        '''
        dealer_points, _ = dealer.get_points()
        player_points, useable_ace = player.get_points()
        if player_points > 21:
            reward = -1
        else:
            if player_points > dealer_points or dealer_points > 21:
                reward = 1
            elif player_points == dealer_points:
                reward = 0
            else:
                reward = -1
        return reward, player_points, dealer_points, useable_ace

    def serve_card_to(self, player, n = 1):
        '''
        Deal cards to the dealer or player,
        if there are not enough cards, shuffle the cards in the open pool and deal again
        Args:
            player A dealer or player
            n The number of consecutive cards dealt at one time
        Return:
            None
        '''
        cards = [] #Cards to be dealt
        for _ in range(n):
            #Consider the situation where the dealer has no cards
            if self.card_q.empty():
                self._info('\n发牌器没牌了，整理废牌，重新洗牌;')
                shuffle(self.cards_in_pool)
                self._info('一共整理了{}张已用牌，重新放入发牌器\n'.format(len(self.cards_in_pool)))
                assert(len(self.cards_in_pool) > 20) #Make sure to collect more cards at once

                #When the code is unreasonably written,
                # there may be continuous bidding even if a certain player bursts,
                # which will cause the player to have more cards in his hand and the dealer and the used cards are few.
                # This situation needs to be avoided.

                self.load_cards(self.cards_in_pool) #Shuffle the collected used cards and send them to the dealer for reuse
            cards.append(self.card_q.get()) # Deal a card from the dealer
        self._info('发了{}张牌({})给{}{}；'.format(n, cards, player.role, player))
        player.receive(cards) #A player accepts the deal
        player.cards_info()

    def _info(self, message):
        if self.display:
            print(message, end='')

    def recycle_cards(self, *players):
        '''
        Recycle the cards in the player's hand to the publicly used card pool
        '''
        if len(players) == 0:
            return
        for player in players:
            for card in player.cards:
                self.cards_in_pool.append(card)
            player.discharge_cards() #Players no longer have these cards in their hands

    def play_game(self, dealer, player):
        '''
        Play a round of 21 points, generate a state sequence and the final reward (the intermediate reward is 0)
        Args:
            dealer/player  dealer and player
        Returns:
            tuple: episode, reward
        '''
        self._info('========= 开始新一局 ========\n')
        self.serve_card_to(player, n=2) #Deal two cards to the player
        self.serve_card_to(dealer, n=2) #Deal two cards to the dealer
        episode = [] # Record a match information
        if player.policy is None:
            self._info('玩家需要一个策略')
            return
        if dealer.policy is None:
            self._info('庄家需要一个策略')
            return
        while True:
            action = player.policy(dealer)
            #The player's strategy produces an action
            self._info('{}{}选择:{};'.format(player.role, player, action))
            episode.append((player.get_state_name(dealer), action)) #Record a(s,a)
            if action == self.A[0]: #Keep bidding
                self.serve_card_to(player) #Deal a card to the player
            else: #Stop bidding
                break
        #After the player stops bidding, the number of points in the player's hand must be counted.
        # If the player bursts, the dealer does not need to continue.
        reward, player_points, dealer_points, useable_ace = self.reward_of(dealer, player)
        if player_points > 21:
            self._info('玩家爆点{}输了，得分:{}\n'.format(player_points, reward))
            self.recycle_cards(player, dealer)
            self.episodes.append((episode, reward)) #When forecasting, you need to form an episode list and focus on learning V
            #When in Monte Carlo control, you don’t need episodes list, generate one episode to learn one, the same below
            self._info('========本局结束========\n')
            return episode, reward
        #The player did not exceed 21 points
        self._info('\n')
        while True:
            action = dealer.policy() #The dealer gets an action from its strategy
            self._info('{}{}选择:{};'.format(dealer.role, dealer, action))
            #The state only records the information of the first card of the dealer,
            # at this time the player no longer bids, (s, a) does not need to be recorded repeatedly
            if action == self.A[0]: #The dealer continues to bid
                self.serve_card_to(dealer)
            else:
                break
        #Both sides stopped bidding
        self._info('\n双方均停止叫牌;\n')
        reward, player_points, dealer_points, useable_ace = self.reward_of(dealer, player)
        player.cards_info()
        dealer.cards_info()
        if reward == +1:
            self._info('玩家赢了!')
        elif reward == -1:
            self._info('玩家输了!')
        else:
            self._info('双方和局!')
        self._info('玩家{}点，庄家{}点\n'.format(player_points,dealer_points))
        self._info('========本局结束========\n')
        self.recycle_cards(player,dealer) #Reclaim the cards in the hands of players and dealers to the open pool
        self.episodes.append((episode, reward)) #Add the complete game just generated to the state sequence list, Monte Carlo control does not need
        return episode, reward

    def play_games(self, dealer, player, num =2, show_statistic = True):
        '''
        Play multiple games at once
        '''
        results = [0, 0, 0] #Player loses, draws, wins
        self.episodes.clear()
        for i in tqdm(range(num)):
            episode, reward = self.play_game(dealer, player)
            results[1+reward] += 1
            if player.learning_method is not None:
                player.learning_method(episode, reward)
        if show_statistic:
            print('共玩了{}局, 玩家赢{}局, 和{}局, 输{}局, 胜率:{:.2f}, 不输率:{:.2f}'\
                .format(num, results[2], results[1], results[0], results[2]/num,(results[2]+results[1])/num))
        return

    def _info(self, message):
        if self.display:
            print(message, end='')






