import numpy as np
from policy_evaluation import evaluate_policy
def policy_improvement(V, P, gamma=1.0):
    """
    Policy improvement algorithm that takes the state-value function of the policy
    you want to improve, V, and the MDP, P (and gamma, optionally).
    
    Args:
        V: state-value function of the current policy
        P: transition matrix (MDP)
        gamma: discount factor (default: 1.0)
    
    Returns:
        new_pi: improved greedy policy
    """
    # can initialize these randomly, but let's keep things simple).
    Q = np.zeros((len(P), len(P[0])), dtype=np.float64)
    
    # (3) Then loop through the states, actions, and transitions.
    for s in range(len(P)):
        for a in range(len(P[s])):
            for prob, next_state, reward, done in P[s][a]:
                # (4) Flag indicating whether next_state is terminal or not
                # (5) We use those values to calculate the Q-function.
                Q[s][a] += prob * (reward + gamma * V[next_state] * (1- done))
    
    # (6) we obtain a new, greedy policy by taking the argmax of the Q-function
    # of the original policy. And there, you have a likely improved policy.
    # Compute the greedy action for each state by taking the argmax over actions
    
    greedy_actions = np.argmax(Q, axis=1) # axis=1 means we take the argmax over the actions for each state
    # Build a mapping from state to greedy action
    state_action_map = {s: a for s, a in enumerate(greedy_actions)}
    # Define the new policy as a function that returns the greedy action for a given state
    def new_pi(s):
        return state_action_map[s]
    
    return new_pi
    

## That's nice , we can can call policy_impovement over and over again to get the optimal policy

def policy_iteration(P,gamma=1.0,theta=1e-6):
    # P: transition matrix
    # gamma: discount factor
    # theta: convergence threshold
    num_states = len(P)
   
    num_actions = len(P[0])
    random_actions = np.random.choice(num_actions, size=num_states)
    pi = lambda s: random_actions[s]
    while True:
        old_pi = {s:pi(s) for s in range(len(P))}
        V = evaluate_policy(pi,P,gamma,theta)
        pi = policy_improvement(V,P,gamma)
        if old_pi == {s:pi(s) for s in range(len(P))}:
            # no change in the policy , we have found the optimal policy
            break
    return pi




