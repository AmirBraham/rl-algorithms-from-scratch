# mc control algorithm implementation
import gymnasium as gym
import numpy as np

# Init Q(s,a) randomly

env = gym.make("FrozenLake-v1", render_mode="human")

n_actions = env.action_space.n
n_states = env.observation_space.n

# INIT Q randomly
Q = np.random.rand(n_states, n_actions)
# Init Pi
Pi = np.argmax(Q, axis=1)
EPSILON = 0.1

gamma = 0.99
NUM_EPISODES = 1000

N = np.zeros((n_states, n_actions))

for episode in range(NUM_EPISODES):
    trajectory = []
    state, _ = env.reset()
    done = False
    while not done:
        if np.random.rand() < EPSILON:
            action = np.random.randint(n_actions)
        else:
            action = Pi[state]
        next_state, reward, done, _, _ = env.step(action)
        trajectory.append((state, action, reward))
        state = next_state
        if done:
            break

    # calculate return:
    G = 0
    first_visit = set()

    for t in range(len(trajectory) - 1, -1, -1):
        state, action, reward = trajectory[t]
        G = gamma * G + reward
        # Check if (S,A) is first visit in trajectory
        if (state, action) not in first_visit:
            first_visit.add((state, action))
            N[state, action] += 1
            Q[state, action] += (G - Q[state, action]) / N[state, action]
    
    # Epsilon greedy update
    for state in range(n_states):
        Pi[state] = np.argmax(Q[state, :])
