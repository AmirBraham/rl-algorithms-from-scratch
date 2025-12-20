

import numpy as np
import gymnasium as gym


def generate_trajectory(pi, env):
    done, trajectory = False, []
    state, _ = env.reset()
    while not done:

        action = pi(state)
        next_state, reward, terminated, truncated, _ = env.step(action)
        done = terminated or truncated
        experience = (state,action,reward,next_state,done)
        trajectory.append(experience)
        state = next_state

    return trajectory

def mc_prediction(pi, env, n_episodes, gamma=1.0, first_visit=True):
    V = np.zeros(env.observation_space.n)
    N = np.zeros(env.observation_space.n)

    for _ in range(n_episodes):
        trajectory = generate_trajectory(pi, env)
        visited_states = set()

        # Calculate returns backwards through the trajectory
        G = 0
        for state, action, reward, next_state, done in reversed(trajectory):
            G = reward + gamma * G

            # First-visit MC: only update if we haven't seen this state yet in this episode
            if first_visit and state in visited_states:
                continue

            visited_states.add(state)
            N[state] += 1
            V[state] = V[state] + (G - V[state]) / N[state]

    return V


def td_prediction(pi, env, n_episodes, alpha=0.1, gamma=1.0):
    """
    TD(0) Prediction - learns value function using temporal difference learning.

    Key difference from MC:
    - MC uses actual return: V(s) <- V(s) + alpha * [G - V(s)]
    - TD uses estimate: V(s) <- V(s) + alpha * [R + gamma*V(s') - V(s)]
                                                 ^^^^^^^^^^^^^^^^^
                                                 TD target (estimated return)

    The TD target (R + gamma*V(s')) is an estimate of G because:
    - We don't wait for the full episode
    - We use the current estimate V(s') instead of the true future return
    - This is called "bootstrapping" - using estimates to update estimates
    """
    V = np.zeros(env.observation_space.n)

    for _ in range(n_episodes):
        state, _ = env.reset()
        done = False

        while not done:
            action = pi(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # TD(0) update rule:
            # V(s) = V(s) + alpha * [reward + gamma * V(next_state) - V(s)]
            #                        ^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^^
            #                        This is the TD error (delta)

            # Hint: What should V(next_state) be if we've reached a terminal state?
            # Think: terminal states have no future return!

            td_target = reward + gamma * V[next_state] * (1 - done)
            td_error = td_target - V[state]
            V[state] = V[state] + alpha * td_error

            state = next_state

    return V


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1", is_slippery=False, render_mode="ansi")

    # Random policy
    def policy(state):
        return np.random.randint(0, 4)  # Random action: 0=LEFT, 1=DOWN, 2=RIGHT, 3=UP


    # Run MC prediction
    V = mc_prediction(policy, env, n_episodes=1000, gamma=0.99, first_visit=True)


    V_td = td_prediction(policy,env,n_episodes=1000, alpha=0.1, gamma=0.99)
    # Display as 4x4 grid (FrozenLake is 4x4)
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            print(f"{V[state]:6.3f}", end=" ")
        print()
    
    print("##########")
     # Display as 4x4 grid (FrozenLake is 4x4)
    for i in range(4):
        for j in range(4):
            state = i * 4 + j
            print(f"{V_td[state]:6.3f}", end=" ")
        print()

    print("\nNote: States with holes (5, 7, 11, 12) and goal (15) should have lower/zero values")
    print("States on the path to goal should have increasing values as we get closer!")
 