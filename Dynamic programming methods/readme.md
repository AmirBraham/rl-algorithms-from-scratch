# Reinforcement Learning Fundamentals: Dynamic Programming Methods

This repository contains implementations of fundamental dynamic programming algorithms for solving Markov Decision Processes (MDPs).

## Overview

The main algorithms implemented here are:
- **Policy Evaluation**: Estimating value functions from a given policy
- **Policy Improvement**: Extracting greedy policies from value functions  
- **Policy Iteration**: Alternating between evaluation and improvement to find optimal policies
- **Value Iteration**: Directly computing optimal value functions and policies

## 1. Policy Evaluation

Policy evaluation is a method for estimating a value function from a policy and an MDP.

### Algorithm
The policy evaluation algorithm iteratively updates the value function using the Bellman equation:

```python
def evaluate_policy(policy, P, gamma=1.0, theta=1e-6):
    # Iteratively compute V(s) for all states
    # until convergence threshold theta is met
```

### Key Features
- Uses the $(1 - \text{done})$ term to handle terminal states correctly
- Prevents infinite loops by properly terminating value propagation at terminal states
- Implements the Bellman equation: $V(s) = \sum_{s'} P(s'|s,\pi(s)) \cdot [R(s,\pi(s),s') + \gamma V(s')]$

## 2. Policy Improvement

Policy improvement is a method for extracting a greedy policy from a value function and an MDP.

### Algorithm
The policy improvement formula is:

$$\pi_{\text{new}}(s) = \arg\max_a \sum_{s', r} p(s', r \mid s, a) \big[ r + \gamma V(s') \big]$$

### Implementation Details
```python
def policy_improvement(V, P, gamma=1.0):
    # Compute Q-function from value function V
    # Return greedy policy: $\pi(s) = \arg\max_a Q(s,a)$
```

### Mathematical Foundation
The Q-function (action-value function) is defined as:

$$Q^{\pi}(s,a) = \sum_{s',r} p(s',r \mid s ,a) \big[ r + \gamma V^{\pi}(s') \big]$$

The greedy policy $\pi'$ satisfies:

$$Q^{\pi'}(s,\pi'(s)) \geq Q^{\pi}(s,\pi(s)) = V^{\pi}(s)$$

This leads to the policy improvement theorem:

$$V^{\pi'}(s) = Q^{\pi'}(s, \pi'(s)) \geq Q^{\pi}(s, \pi'(s)) \geq V^{\pi}(s)$$

## 3. Policy Iteration

Policy iteration consists of alternating between policy evaluation and policy improvement to obtain an optimal policy from an MDP.

### Algorithm
```python
def policy_iteration(P, gamma=1.0, theta=1e-6):
    # Start with random policy
    # Repeat until policy converges:
    #   1. Evaluate current policy (get V)
    #   2. Improve policy using V
    # Return optimal policy
```

### Convergence
The algorithm converges when the policy stops changing between iterations, indicating that an optimal policy has been found.

## 4. Value Iteration

Value iteration is mathematically similar to iterative policy evaluation but directly computes optimal value functions.

### Algorithm
The value iteration update equation is:

$$V_{k+1}(s) = \max_a \sum_{s', r} p(s', r \mid s, a) \left[ r + \gamma V_k(s') \right]$$

### Key Differences from Policy Iteration
- **Value Iteration**: Directly optimizes the value function using the Bellman optimality equation
- **Policy Iteration**: Alternates between policy evaluation and improvement

### Mathematical Components
- $V_{k+1}(s)$: Updated value for state s at iteration k+1
- $\max_a$: Maximization over all possible actions a
- $p(s',r|s,a)$: Transition probability from state s to s' with reward r under action a
- $\gamma$: Discount factor
- $V_k(s')$: Value of next state at iteration k

## Implementation Notes

### Terminal State Handling
All algorithms use the $(1 - \text{done})$ term to properly handle terminal states:
- For non-terminal states: $(1 - \text{done}) = 1$ → normal value propagation
- For terminal states: $(1 - \text{done}) = 0$ → only immediate rewards, no future value

### Convergence Criteria
- **Policy Evaluation**: Uses $\theta$ threshold to determine when value function has converged
- **Policy Iteration**: Stops when policy stops changing between iterations
- **Value Iteration**: Uses similar convergence criteria as policy evaluation

## Usage

```python
# Example usage of policy iteration
from policy_evaluation import evaluate_policy
from policy_enhancement import policy_iteration

# Define your MDP transition matrix P
# Run policy iteration to find optimal policy
optimal_policy = policy_iteration(P, gamma=0.9, theta=1e-6)
```

## Files

- `policy_evaluation.py`: Implementation of policy evaluation algorithm
- `policy_enhancement.py`: Implementation of policy improvement and policy iteration
- `readme.md`: This comprehensive documentation file 