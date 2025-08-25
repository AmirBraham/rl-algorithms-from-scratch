# Yes, this is a correct implementation of the value iteration algorithm for MDPs.
# It iteratively updates the value function for each state by taking the maximum expected value over all possible actions,
# using the Bellman optimality equation, until the value function converges within a specified threshold (theta).
import numpy as np
def value_iteration(P, gamma=1.0, theta=1e-6):
    """
    Value Iteration algorithm for solving MDPs.

    Args:
        P: transition matrix (P[s][a] is a list of (prob, next_state, reward, done))
        gamma: discount factor
        theta: convergence threshold

    Returns:
        V: optimal state-value function
    """
    num_states = len(P)
    V = np.zeros(num_states)
    
    while True:
        delta = 0
        for s in range(num_states):
            v = V[s]
            action_values = []
            for a in range(len(P[s])):
                q_sa = 0
                for prob, next_state, reward, done in P[s][a]:
                    if done:
                        q_sa += prob * reward
                    else:
                        q_sa += prob * (reward + gamma * V[next_state])
                action_values.append(q_sa)
            V[s] = max(action_values)
            delta = max(delta, abs(v - V[s]))
        if delta < theta:
            break

    # Extract optimal policy from value function
    def pi(s):
        action_values = []
        for a in range(len(P[s])):
            q_sa = 0
            for prob, next_state, reward, done in P[s][a]:
                if done:
                    q_sa += prob * reward
                else:
                    q_sa += prob * (reward + gamma * V[next_state])
            action_values.append(q_sa)
        return np.argmax(action_values)

    return V, pi