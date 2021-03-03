def str_key(*args):
    '''
    Connect the parameters with'_' as the keys of the dictionary.
    Note that the parameters themselves may be tuple or list types,
    such as the form similar to ((a,b,c),d)
    '''
    new_arg = []
    for arg in args:
        if type(arg) in [tuple,list]:
            new_arg += [str(i) for i in arg]
        else:
            new_arg.append(str(arg))
    return "_".join(new_arg)

def set_dict(target_dict,value,*args):
    target_dict[str_key(*args)] = value

def set_prob(P, s, a, s1, p = 1.0): #Set up probability dictionary
    set_dict(P, p, s, a, s1)

def get_prob(P, s, a, s1): #get probability value
    return P.get(str_key(s, a, s1),0)

def set_reward(R, s, a, r): #Set up reward value
    set_dict(R, r, s, a)

def get_reward(R, s, a): # get reward value
    return R.get(str_key(s,a),0)

def display_dict(target_dict): #Show dictionary contents
    for key in target_dict.keys():
        print('{}:  {:.2f}'.format(key,target_dict[key]))
    print('')

def set_value(V, s, v): # Set up value dictionary
    set_dict(V, v, s)

def get_value(V, s): #get value dictionary
    return V.get(str_key(s),0)

def set_pi(Pi, s, a, p = 0.5): #Set up policy dictionary
    set_dict(Pi, p, s, a)

def get_pi(Pi, s, a): # get policy (probability) value
    return Pi.get(str_key(s,a),0)