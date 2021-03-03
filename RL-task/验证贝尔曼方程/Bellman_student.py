#Import tool function: generate operation-related dictionary keys according to state and behavior, and display dictionary contents
import os
import sys
path = os.path.dirname(__file__)
sys.path.append(path)
from utils import str_key,display_dict
#Set transition probability, reward value and reading method
from utils import set_prob,set_reward,get_prob,get_reward
#Set status value, strategy probability, and reading method
from utils import set_value,set_pi,get_value,get_pi

#Constructing Student Markov Decision Process
S = ['浏览手机中','第一节课','第二节课','第三节课','休息中']
A = ['浏览手机','学习','离开浏览','泡吧','退出学习']
R = {} #Reward Rsa dictionary
P = {} #State transition probability Pss'a dictionary
gamma = 1.0 #Attenuation factor

#Set the state transition probability and reward according to the data of the student Markov decision process example, the default probability is 1
set_prob(P, S[0], A[0], S[0]) #浏览手机中 - 浏览手机 -> 浏览手机中
set_prob(P, S[0], A[2], S[1]) #浏览手机中 - 离开浏览 -> 第一节课
set_prob(P, S[1], A[0], S[0])#第一节课 - 浏览手机 -> 浏览手机中
set_prob(P, S[1], A[1], S[2])#第一节课 - 学习 -> 第二节课
set_prob(P, S[2], A[1], S[3])#第二节课 - 学习 —> 第三节课
set_prob(P, S[2], A[4], S[4])#第二节课 - 退出学习 -> 休息中
set_prob(P, S[3], A[1], S[4])#第三节课 - 学习 -> 休息中
set_prob(P, S[3], A[3], S[1], p = 0.2)#第三节课 - 泡吧 -> 第一节课
set_prob(P, S[3], A[3], S[2], p = 0.4)#第三节课 - 泡吧 -> 第二节课
set_prob(P, S[3], A[3], S[3], p = 0.4)#第三节课 - 泡吧 -> 第三节课

set_reward(R, S[0], A[0], -1)#浏览手机中 - 浏览手机 -> -1
set_reward(R, S[0], A[2], 0)#浏览手机中 - 离开浏览 -> 0
set_reward(R, S[1], A[0], -1)#第一节课 - 浏览手机 -> -1
set_reward(R, S[1], A[1], -2)#第一节课 - 学习 -> -2
set_reward(R, S[2], A[1], -2)#第二节课 - 学习 -> -2
set_reward(R, S[2], A[4], 0)#第二节课 - 退出学习 -> 0
set_reward(R, S[3], A[1], 10)#第三节课 - 学习 -> 10
set_reward(R, S[3], A[3], +1)#第三节课 - 泡吧 -> +1
MDP = (S, A, R, P, gamma)
print('----状态转移概率字典（矩阵）信息:----')
display_dict(P)
print('---奖励字典(函数)信息:----')
display_dict(R)

Pi = {} # Strategy Dictionary
set_pi(Pi, S[0], A[0], 0.5)#浏览手机中 - 浏览手机
set_pi(Pi, S[0], A[2], 0.5)#浏览手机中 - 离开浏览
set_pi(Pi, S[1], A[0], 0.5)#第一节课 - 浏览手机
set_pi(Pi, S[1], A[1], 0.5)#第一节课 - 学习
set_pi(Pi, S[2], A[1], 0.5)#第二节课 - 学习
set_pi(Pi, S[2], A[4], 0.5)#第二节课 - 退出学习
set_pi(Pi, S[3], A[1], 0.5)#第三节课 - 学习
set_pi(Pi, S[3], A[3], 0.5)#第三节课 - 泡吧


print('----状态转移概率字典（矩阵）信息:----')
display_dict(Pi)
print('---奖励字典(函数)信息:----')
V = {} #Value Dictionary
display_dict(V)

def compute_q(MDP, V, s, a):
    '''
    According to the given MDP, the value function V, calculate the value qsa of the state behavior to s, a
    formula   $$q_{\pi}(s,a) = R^a_b + \gamma \sum_{s' \in S}P^a_{ss'} \nu \pi(s')  $$    #markdown
    '''
    S, A, R, P, gamma = MDP
    q_sa = 0
    for s_prime in S:
        q_sa += get_prob(P, s, a, s_prime) * get_value(V, s_prime)
        q_sa =  get_reward(R, s, a) + gamma * q_sa
    return q_sa

def compute_v(MDP, V, Pi, s):
    '''
    Calculate the value of state s according to a certain strategy Pi and the current state value function V under a given MDP
    formula   $$ \nu_\pi(s) = \sum_{a \in A} \pi(a|s)q_\pi(s,a) $$     # markdown
    '''
    S, A, R, P, gamma = MDP
    v_s = 0
    for a in A:
        v_s += get_pi(Pi, s, a) * compute_q(MDP, V, s, a)

    return v_s

#Use the retrospective method to update the state value according to the current strategy. Chapter 3 will talk about
def update_V(MDP, V, Pi):
    '''
    Given an MDP and a strategy, update the value function V under the strategy
    '''
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        #set_value(V_prime,s,compute_v(MDP, V, Pi, s))
        V_prime[str_key(s)] = compute_v(MDP, V_prime, Pi, s)
    return V_prime

#Strategy evaluation, get the final state value under the strategy
def policy_evaluate(MDP, V, Pi, n):
    '''
    Use n iterations of calculation to evaluate the state value of an MDP under a given strategy Pi, the initial value is V
    '''
    for i in range(n):
        V = update_V(MDP, V, Pi)
    return V

V = policy_evaluate(MDP, V, Pi, 1)
display_dict(V)

v = compute_v(MDP, V, Pi, '第三节课')
print('第三节课在当前从策略下的最终价值为:{:.2f}'.format(v))

def compute_v_from_max_q(MDP, V, s):
    '''
    Determine the current state value based on the largest of all possible behavioral values in a state
    '''
    S, A, R, P, gamma = MDP
    v_s = -float('inf')
    for a in A:
        qsa = compute_q(MDP, V, s, a)
        if qsa >= v_s:
            v_s = qsa
    return v_s

def update_V_without_pi(MDP, V):
    '''
    Update the state value directly through the value of the subsequent state without relying on the strategy
    '''
    S, _, _, _, _ = MDP
    V_prime = V.copy()
    for s in S:
        V_prime[str_key(s)] = compute_v_from_max_q(MDP, V_prime, s)
    return V_prime

#Value iteration
def value_iterate(MDP, V, n):
    '''
    Value iteration
    '''
    for i in range(n):
        V = update_V_without_pi(MDP, V)
    return V
V = {}
#Obtain the optimal state value through value iteration
V_star = value_iterate(MDP, V, 4)
display_dict(V_star)

s, a = '第三节课','泡吧'
q = compute_q(MDP, V_star, '第三节课','泡吧')
print('在状态{}选择行为{}的最优价值为:{:.2f}'.format(s, a, q))

