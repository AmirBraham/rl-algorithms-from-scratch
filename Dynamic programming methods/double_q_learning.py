# double q learning algorithm implementation
import gymnasium as gym
from gymnasium.wrappers import RecordVideo, RecordEpisodeStatistics
import numpy as np


def double_q_learning(
    env,
    gamma=1.0,
    init_alpha=0.5,
    min_alpha=0.01,
    alpha_decay_ratio=0.5,
    init_epsilon=1.0,
    min_epsilon=0.1,
    epsilon_decay_ratio=0.9,
    n_episodes=3000,
):
    alpha = init_alpha
    epsilon = init_epsilon
    n_actions = env.action_space.n
    n_states = env.observation_space.n
    Q1 = np.zeros((n_states, n_actions))
    Q2 = np.zeros((n_states, n_actions))
    
    # Calculate decay rates per episode
    alpha_decay = (min_alpha / init_alpha) ** (1 / (n_episodes * alpha_decay_ratio))
    epsilon_decay = (min_epsilon / init_epsilon) ** (1 / (n_episodes * epsilon_decay_ratio))
    
    episode_rewards = []
    episode_lengths = []
    success_count = 0

    for episode in range(n_episodes):
        state, _ = env.reset()
        done = False
        episode_reward = 0
        episode_length = 0
        
        while not done:
            if np.random.uniform() < epsilon:
                action = np.random.randint(n_actions)
            else:
                action = np.argmax((Q1[state] + Q2[state]) / 2)
            next_state, reward, done, _, _ = env.step(action)
            episode_reward += reward
            episode_length += 1

            # we can alternate between using Q1 and Q2 in this tabular setting
            # FOR double DQN, we use Q1 to select the action and Q2 to estimate the value of the action
            if np.random.uniform() < 0.5:
                # Handle terminal states: if done, no future value
                if done:
                    target = reward
                else:
                    target = reward + gamma * Q2[next_state, np.argmax(Q1[next_state])]
                Q1[state, action] = Q1[state, action] + alpha * (target - Q1[state, action])
            else:
                # Handle terminal states: if done, no future value
                if done:
                    target = reward
                else:
                    target = reward + gamma * Q1[next_state, np.argmax(Q2[next_state])]
                Q2[state, action] = Q2[state, action] + alpha * (target - Q2[state, action])
            
            state = next_state
        
        # Decay per episode, not per step
        alpha = max(min_alpha, alpha * alpha_decay)
        epsilon = max(min_epsilon, epsilon * epsilon_decay)
        
        episode_rewards.append(episode_reward)
        episode_lengths.append(episode_length)
        if episode_reward > 0:  # Success in FrozenLake
            success_count += 1
        
    
    total_success_rate = success_count / n_episodes
    print(f"\nTraining complete! Total success rate: {total_success_rate:.2%}")
    print(f"Final success rate (last 100 episodes): {sum(episode_rewards[-100:]) / 100.0:.2%}")
    
    return Q1, Q2


if __name__ == "__main__":
    env = gym.make("FrozenLake-v1",  render_mode="rgb_array")
    
    num_eval_episodes = 100
    
    # Add video recording for every episode
    env = RecordVideo(
        env,
        video_folder="cartpole-agent",    # Folder to save videos
        name_prefix="eval",               # Prefix for video filenames
        episode_trigger=lambda x: x % 200 == 0    # Record every episode
    )
    
    # Add episode statistics tracking
    env = RecordEpisodeStatistics(env, buffer_length=num_eval_episodes)
    
    try:
        # Run double Q-learning
        Q1, Q2 = double_q_learning(env, n_episodes=10_000)
    finally:
        # Ensure proper cleanup
        env.close()
