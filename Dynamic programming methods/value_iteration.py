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
    Q = np.zeros((num_states, len(P[0])))
    while True:
        prev_V = V.copy()
        for s in range(num_states):
            action_values = []
            for a in range(len(P[s])):
                q_sa = 0
                for prob, next_state, reward, done in P[s][a]:
                    Q[s][a] += prob * (reward + gamma * prev_V[next_state] * (1 - done))
                action_values.append(Q[s][a])
            V[s] = max(action_values)
        delta = max(abs(V - prev_V))
        if delta < theta:
            break

    # We can extract pi from V by taking the argmax of the Q-function
    pi = lambda s: {s:a for s,a in enumerate(np.argmax(Q,axis=1))}[s]

    return V,pi