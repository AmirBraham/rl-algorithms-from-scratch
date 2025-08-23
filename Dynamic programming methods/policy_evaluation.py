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
    prev_V = np.zeros(number_of_states) # V_k-1
    while True:
        for s in range(number_of_states):
            for prob,next_state,reward,done in P[s][policy(s)]:
                V[s]+=(reward + gamma * prev_V[next_state] * (1 - done))*prob
                # we multiply by (1-done) to avoid updating V[terminal] 
                # else we end up with V[terminal] = reward + gamma * V[next_state] where next_State can only be the terminal state or undefined
        delta = max(abs(V-prev_V))
        if delta < theta:
            break
        prev_V = V.copy()
    return V