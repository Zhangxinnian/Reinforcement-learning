import matplotlib.pyplot as plt
import numpy as np
import random  # 随机策略时用到


def str_key(*args):
    '''
    Connect the parameters with'_' as the keys of the dictionary.
    Note that the parameters themselves may be tuple or list types,
    such as the form similar to ((a,b,c),d)
    '''
    new_arg = []
    for arg in args:
        if type(arg) in [tuple, list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)


def set_dict(target_dict, value, *args):
    target_dict[str_key(*args)] = value


def get_dict(target_dict, *args):
    return target_dict.get(str_key(*args), 0)


def greedy_pi(A, s, Q, a):
    '''
    According to greedy selection, calculate the probability of behavior a being greedily selected in behavior space A in state s
     Consider a situation where multiple actions have the same value
    '''
    # print("in greedy_pi: s={},a={}".format(s,a))
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:  # Count the maximum value of the subsequent state and the behavior of reaching that state (maybe more than one)
        q = get_dict(Q, s, a_opt)
        # print("get q from dict Q:{}".format(q))
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            # print("in greedy_pi: {} == {}".format(q,max_q))
            a_max_q.append(a_opt)
    n = len(a_max_q)
    if n == 0: return 0.0
    return 1.0 / n if a in a_max_q else 0.0


def greedy_policy(A, s, Q):
    """
    In a given state, select a behavior a from the behavior space A such that Q(s,a) = max(Q(s,))
     Consider the situation where multiple behaviors have the same value
    """
    max_q, a_max_q = -float('inf'), []
    for a_opt in A:
        q = get_dict(Q, s, a_opt)
        if q > max_q:
            max_q = q
            a_max_q = [a_opt]
        elif q == max_q:
            a_max_q.append(a_opt)
    return random.choice(a_max_q)


def epsilon_greedy_pi(A, s, Q, a, epsilon=0.1):
    m = len(A)
    greedy_p = greedy_pi(A, s, Q, a)
    # print("greedy prob:{}".format(greedy_p))
    if greedy_p == 0:
        return epsilon / m
    n = int(1.0 / greedy_p)
    return (1 - epsilon) * greedy_p + epsilon / m


def epsilon_greedy_policy(A, s, Q, epsilon, show_random_num=False):
    pis = []
    m = len(A)
    for i in range(m):
        pis.append(epsilon_greedy_pi(A, s, Q, A[i], epsilon))
    rand_value = random.random()
    # if show_random_num:
    #    print("产生的随机数概率为:{:.2f}".format(rand_value))
    # print(rand_value)
    for i in range(m):
        if show_random_num:
            print("随机数:{:.2f}, 拟减去概率{}".format(rand_value, pis[i]))
        rand_value -= pis[i]
        if rand_value < 0:
            return A[i]

