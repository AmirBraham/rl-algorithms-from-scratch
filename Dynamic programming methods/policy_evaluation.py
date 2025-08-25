# policy evaluation is an iterative process to get the the true value function of a policy
import numpy as np

def evaluate_policy(policy,P,gamma=1.0,theta=1e-6):
    # policy: a function that maps states to actions
    # P: transition matrix ( probability of next state given current state and action)
    # P[s][policy(s)] -> prob , next_state, reward , done : probability of next state given current state and action given by the policy
    # gamma: discount factor
    # theta: convergence threshold since this is an iterative process
    number_of_states = len(P)
    V = np.zeros(number_of_states) # value function is a list of length of the number of states   V_k
    
    while True:
        delta = 0
        for s in range(number_of_states):
            v = V[s]
            V[s] = 0
            for prob, next_state, reward, done in P[s][policy(s)]:
                if done:
                    V[s] += prob * reward
                else:
                    V[s] += prob * (reward + gamma * V[next_state])
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break
    return V