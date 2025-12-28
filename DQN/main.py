import math
import random
from collections import deque, namedtuple
from itertools import count

import gymnasium as gym
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

env = gym.make("CartPole-v1")
eval_env = gym.make("CartPole-v1", render_mode="human")

device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

Transition = namedtuple("Transition", ("state", "action", "next_state", "reward"))


class ReplayMemory(object):
    def __init__(self, N) -> None:
        self.memory = deque([], maxlen=N)

    def push(self, transition:Transition):
        self.memory.append(transition)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQN(nn.Module):
    def __init__(self, n_observations, n_actions):
        super(DQN, self).__init__()
        self.layer1 = nn.Linear(n_observations, 256)
        self.layer2 = nn.Linear(256, 256)
        self.layer3 = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.layer1(x))
        x = F.relu(self.layer2(x))
        x = self.layer3(x)
        return x


# Batch size
BATCH_SIZE = 128

# Probability epsilon of choosing a random action
EPSILON = 0.95
EPSILON_DECAY = 0.995
EPSILON_MIN = 0.01

# Discount factor
GAMMA = 0.99

# Gradient descent learning rate
LEARNING_RATE = 0.001


n_actions = env.action_space.n
state, info = env.reset()
n_observations = len(state)

policy_net = DQN(n_observations, n_actions).to(device)
target_net = DQN(n_observations, n_actions).to(device)
target_net.load_state_dict(policy_net.state_dict())


optimizer = optim.AdamW(policy_net.parameters(), lr=LEARNING_RATE)
CAPACITY = 10_000
memory = ReplayMemory(CAPACITY)

TAU = 0.005

steps_done = 0


def select_action(state):
    global steps_done
    sample = random.random()
    eps_threshold = EPSILON_MIN + (EPSILON - EPSILON_MIN) * math.exp(
        -1.0 * steps_done / EPSILON_DECAY
    )
    steps_done += 1
    if sample > eps_threshold:
        with torch.no_grad():
            return policy_net(state).max(1).indices.view(1, 1)
    else:
        return torch.tensor(
            [[env.action_space.sample()]], device=device, dtype=torch.long
        )

## Training Loop


def play_episode():
    """Play one episode with rendering using the current policy (greedy)."""
    state, _ = eval_env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    total_reward = 0.0
    steps = 0
    for t in count():
        # Always use greedy action (no exploration)
        with torch.no_grad():
            action = policy_net(state).max(1).indices.view(1, 1)
        observation, reward, terminated, truncated, _ = eval_env.step(action.item())
        total_reward += float(reward)
        steps = t + 1
        done = terminated or truncated

        if not done:
            state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)
        else:
            break

    return total_reward, steps


def optimize_model():
    transitions = memory.sample(BATCH_SIZE)
    batch = Transition(*zip(*transitions))

    non_final_mask = torch.tensor(
        tuple(map(lambda s: s is not None, batch.next_state)), device=device, dtype=torch.bool
    )

    non_final_next_states = torch.cat([s for s in batch.next_state if s is not None])
    state_batch = torch.cat(batch.state)
    action_batch = torch.cat(batch.action)
    reward_batch = torch.cat(batch.reward)

    state_action_values = policy_net(state_batch).gather(1, action_batch)

    # Compute V(s_{t+1}) for all next states.
    # Expected values of actions for non_final_next_states are computed based
    # on the "older" target_net; selecting their best reward with max(1).values
    # This is merged based on the mask, such that we'll have either the expected
    # state value or 0 in case the state was final.
    next_state_values = torch.zeros(BATCH_SIZE, device=device)
    with torch.no_grad():
        next_state_values[non_final_mask] = target_net(non_final_next_states).max(1).values
    # Compute the expected Q values
    expected_state_action_values = (next_state_values * GAMMA) + reward_batch # this is our target

    criterion = nn.SmoothL1Loss()
    loss = criterion(state_action_values, expected_state_action_values.unsqueeze(1))
    optimizer.zero_grad()
    loss.backward()

    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()


num_episodes = 600

for i_episode in range(num_episodes):
    # Initialize the environment and get its state
    state, info = env.reset()
    state = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
    for t in count():
        action = select_action(state) # based on probability epsilon
        observation, reward, terminated, truncated, _ = env.step(action.item())
        reward = torch.tensor([reward], device=device)
        done = terminated or truncated

        if terminated:
            next_state = None
        else:
            next_state = torch.tensor(observation, dtype=torch.float32, device=device).unsqueeze(0)


        # Store the transition in memory
        memory.push(Transition(state, action, next_state, reward))

        # Move to the next state
        state = next_state

        # Perform one step of the optimization (on the policy network)
        if len(memory) >= BATCH_SIZE:
            optimize_model()

        # Soft update of the target network's weights
        # θ′ ← τ θ + (1 −τ )θ′
        target_net_state_dict = target_net.state_dict()
        policy_net_state_dict = policy_net.state_dict()
        for key in policy_net_state_dict:
            target_net_state_dict[key] = policy_net_state_dict[key]*TAU + target_net_state_dict[key]*(1-TAU)
        target_net.load_state_dict(target_net_state_dict)

        if done:
            break

    # Play a rendered episode every 50 episodes
    if (i_episode + 1) % 50 == 0:
        total_reward, steps = play_episode()
        print(f"Episode {i_episode + 1}: Played episode with {steps} steps and total reward {total_reward}")

print('Complete')