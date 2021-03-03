import numpy as np


#student states  dict {'0':'C1','1':'C2','2':'C3','3':'pass','4':'Pub','5':'FB','6':'Sleep'}
num_states = 7
i_to_n = {}    #Dictionary indexed to state names
i_to_n['0'] = 'C1'
i_to_n['1'] = 'C2'
i_to_n['2'] = 'C3'
i_to_n['3'] = 'Pass'
i_to_n['4'] = 'Pub'
i_to_n['5'] = 'FB'
i_to_n['6'] = 'Sleep'

n_to_i = {}  #Dictionary state names to indexed
for i,name in zip(i_to_n.keys(),i_to_n.values()):
    n_to_i[name] = int(i)

#State transition probability matrix    C1  C2  C3  Pass Pub  FB  Sleep
Pss = [
    [0.0,0.5,0.0,0.0,0.0,0.5,0.0],
    [0.0,0.0,0.8,0.0,0.0,0.0,0.2],
    [0.0,0.0,0.0,0.6,0.4,0.0,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,1.0],
    [0.2,0.4,0.4,0.0,0.0,0.0,0.0],
    [0.1,0.0,0.0,0.0,0.0,0.9,0.0],
    [0.0,0.0,0.0,0.0,0.0,0.0,1.0]
]
Pss = np.array(Pss)
#Reward function, respectively corresponding to the state
rewards = [-2,-2,-2,10,1,-1,0]
gamma = 0.5 #Attenuation factor

def compute_return(start_index = 0,chain = None,gamma = 0.5) -> float:
    '''Calculate the harvest value of a state in a Markov reward process
    Args:
        start_index The position in the chain of the state to be calculated
        chain Markov process to be calculated
        gamma Attenuation factor
    Returns:
        retrn Harvest value
    '''
    retrn ,power,gamma = 0.0,0,gamma
    for i in range(start_index,len(chain)):
        retrn += np.power(gamma,power) * rewards[n_to_i[chain[i]]]
        power += 1
    return retrn

#Define 4 Markov chains starting with S1
chains = [
    ['C1','C2','C3','Pass','Sleep'],
    ['C1','FB','FB','C1','C2','Sleep'],
    ['C1','C2','C3','Pub','C2','C3','Pass','Sleep'],
    ['C1','FB','FB','C1','C2','C3','Pub','C1','FB','FB','FB','C1','C2','C3','Pub','C2','Sleep']
]
#Verify the harvest value of the starting state of the last Markov chain
value = compute_return(0,chains[3],gamma= 0.5)
print(value)

def compute_value(Pss,rewards,gamma = 0.05):
    '''Calculate the value of the state directly by solving the matrix equation
    Args:
        P State transition probability matrix shape(7,7)
        rewards Instant reward list
        gamma Attenuation factor
    Returns:
        values State value
    '''
    #assert(gamma >= 0 and gamma <= 1.0)
    #Convert rewards to numpy array and modify it to the form of column vector
    rewards = np.array(rewards).reshape((-1,1))
    #np.eye(7,7) is the identity matrix, and the inv method is to find the inverse of the matrix
    values = np.dot(np.linalg.inv(np.eye(7,7) - gamma * Pss),rewards)
    return values

values = compute_value(Pss,rewards,gamma=0.99999)
print(values)
