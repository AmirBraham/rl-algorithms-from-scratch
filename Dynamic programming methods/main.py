import numpy as np
from policy_evaluation import evaluate_policy
from value_iteration import value_iteration
from policy_enhancement import policy_iteration
import gymnasium as gym
import matplotlib.pyplot as plt

env = gym.make('FrozenLake-v1', render_mode=None)
P = env.unwrapped.P

# FrozenLake 4x4 map layout
lake_map = [
    "SFFF",
    "FHFH", 
    "FFFH",
    "HFFG"
]

gamma_values = [0.5, 0.8, 0.9, 0.95, 0.99]

# Store results for plotting
results = {
    'gamma': [],
    'simple_policy_values': [],
    'optimal_values': [],
    'policy_iteration_values': []
}

for gamma in gamma_values:
    def simple_policy(state):
        return 2 # 2 means to always move right
    
    V_simple = evaluate_policy(simple_policy, P, gamma=gamma, theta=1e-6)
    V_opt, pi_opt = value_iteration(P, gamma=gamma, theta=1e-6)
    pi_policy_iter = policy_iteration(P, gamma=gamma, theta=1e-6)
    V_policy_iter = evaluate_policy(pi_policy_iter, P, gamma=gamma, theta=1e-6)
    
    # Store results
    results['gamma'].append(gamma)
    results['simple_policy_values'].append(np.mean(V_simple))
    results['optimal_values'].append(np.mean(V_opt))
    results['policy_iteration_values'].append(np.mean(V_policy_iter))

env.close()

# Create visualizations
plt.figure(figsize=(18, 12))

# Plot 1: Average value vs gamma
plt.subplot(2, 4, 1)
plt.plot(results['gamma'], results['simple_policy_values'], 'o-', label='Simple Policy (Right)', linewidth=2)
plt.plot(results['gamma'], results['optimal_values'], 's-', label='Value Iteration', linewidth=2)
plt.plot(results['gamma'], results['policy_iteration_values'], '^-', label='Policy Iteration', linewidth=2)
plt.xlabel('Discount Factor (γ)')
plt.ylabel('Average Value')
plt.title('Policy Performance vs Discount Factor')
plt.legend()
plt.grid(True, alpha=0.3)

# Plot 2: Value function heatmap for gamma=0.9
plt.subplot(2, 4, 2)
V_opt_09 = value_iteration(P, gamma=0.9, theta=1e-6)[0]
V_grid = V_opt_09.reshape(4, 4)
im = plt.imshow(V_grid, cmap='viridis', interpolation='nearest')
plt.colorbar(im)
plt.title('Value Function Heatmap (γ=0.9)')
plt.xticks(range(4))
plt.yticks(range(4))

# Add state numbers to heatmap
for i in range(4):
    for j in range(4):
        state = i * 4 + j
        plt.text(j, i, f'{state}\n{V_grid[i,j]:.3f}', ha='center', va='center', 
                fontsize=8, fontweight='bold', color='white')

# Plot 3: Policy visualization with map layout for gamma=0.9
plt.subplot(2, 4, 3)
pi_opt_09 = value_iteration(P, gamma=0.9, theta=1e-6)[1]
policy_grid = np.array([pi_opt_09(s) for s in range(16)]).reshape(4, 4)

# Create background colors based on map layout
background_colors = np.zeros((4, 4, 4))  # RGBA
for i in range(4):
    for j in range(4):
        tile = lake_map[i][j]
        if tile == 'S':  # Start - light blue
            background_colors[i, j] = [0.7, 0.9, 1.0, 0.8]
        elif tile == 'G':  # Goal - green
            background_colors[i, j] = [0.7, 1.0, 0.7, 0.8]
        elif tile == 'H':  # Hole - red
            background_colors[i, j] = [1.0, 0.7, 0.7, 0.8]
        else:  # Frozen - white
            background_colors[i, j] = [1.0, 1.0, 1.0, 0.8]

plt.imshow(background_colors)
action_names = ['←', '↓', '→', '↑']
action_colors = ['red', 'blue', 'green', 'orange']

for i in range(4):
    for j in range(4):
        state = i * 4 + j
        action = policy_grid[i, j]
        tile = lake_map[i][j]
        
        # Show tile type and state number
        plt.text(j, i, f'{tile}\n{state}', ha='center', va='center', fontsize=10, fontweight='bold')
        
        # Show policy arrow (only for non-terminal states)
        if tile not in ['H', 'G']:
            plt.text(j, i, action_names[action], ha='center', va='center', 
                    fontsize=14, fontweight='bold', color=action_colors[action],
                    bbox=dict(boxstyle="circle,pad=0.2", facecolor='white', alpha=0.8))

plt.title('Optimal Policy (γ=0.9)\nS=Start, G=Goal, H=Hole')
plt.xticks(range(4))
plt.yticks(range(4))

# Plot 4: Detailed value function across all 16 states
plt.subplot(2, 4, 4)
states = list(range(16))
V_opt_09 = value_iteration(P, gamma=0.9, theta=1e-6)[0]
V_simple_09 = evaluate_policy(lambda s: 2, P, gamma=0.9, theta=1e-6)

plt.bar([s-0.2 for s in states], V_simple_09, width=0.4, alpha=0.7, label='Simple Policy (Right)', color='orange')
plt.bar([s+0.2 for s in states], V_opt_09, width=0.4, alpha=0.7, label='Optimal Policy', color='blue')
plt.xlabel('State Number')
plt.ylabel('Value')
plt.title('Value Function Comparison (γ=0.9)')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(states)

# Plot 5: Value function comparison across different gammas
plt.subplot(2, 4, 5)
for gamma in [0.5, 0.8, 0.9, 0.99]:
    V_gamma = value_iteration(P, gamma=gamma, theta=1e-6)[0]
    plt.plot(states, V_gamma, 'o-', label=f'γ={gamma}', linewidth=2, markersize=4)
plt.xlabel('State')
plt.ylabel('Value')
plt.title('Value Functions Across States')
plt.legend()
plt.grid(True, alpha=0.3)
plt.xticks(states)

# Plot 6: State-by-state improvement
plt.subplot(2, 4, 6)
improvement_per_state = V_opt_09 - V_simple_09
colors = ['green' if imp > 0 else 'red' for imp in improvement_per_state]
plt.bar(states, improvement_per_state, color=colors, alpha=0.7)
plt.xlabel('State')
plt.ylabel('Improvement')
plt.title('Optimal vs Simple Policy\nImprovement per State')
plt.grid(True, alpha=0.3)
plt.xticks(states)

# Plot 7: Convergence comparison
plt.subplot(2, 4, 7)
iterations = [1, 2, 5, 10, 20, 50, 100]
convergence_values = []
for iters in iterations:
    V = np.zeros(16)
    for _ in range(iters):
        for s in range(16):
            v = V[s]
            V[s] = 0
            for prob, next_state, reward, done in P[s][2]:  # Simple policy (action 2)
                if done:
                    V[s] += prob * reward
                else:
                    V[s] += prob * (reward + 0.9 * V[next_state])
    convergence_values.append(np.mean(V))

plt.plot(iterations, convergence_values, 'o-', linewidth=2)
plt.xlabel('Iterations')
plt.ylabel('Average Value')
plt.title('Policy Evaluation Convergence')
plt.grid(True, alpha=0.3)
plt.xscale('log')

# Plot 8: Performance improvement across gammas
plt.subplot(2, 4, 8)
improvement = np.array(results['optimal_values']) - np.array(results['simple_policy_values'])
plt.bar(range(len(gamma_values)), improvement, alpha=0.7, color='purple')
plt.xlabel('Gamma Index')
plt.ylabel('Improvement')
plt.title('Optimal vs Simple Policy\nImprovement')
plt.xticks(range(len(gamma_values)), [f'γ={g}' for g in gamma_values])
plt.grid(True, alpha=0.3)

plt.tight_layout()
plt.savefig('policy_analysis.png', dpi=300, bbox_inches='tight')
plt.show()

print("Visualization saved as 'policy_analysis.png'")
print(f"Results summary:")
print(f"Best performing gamma: {gamma_values[np.argmax(results['optimal_values'])]}")
print(f"Maximum average value: {max(results['optimal_values']):.4f}")
print(f"\nFrozenLake Map Layout (states 0-15):")
for i, row in enumerate(lake_map):
    state_nums = [str(i*4 + j) for j in range(4)]
    print(f"  {row}  States: {state_nums}")
print(f"\nValue function for γ=0.9:")
for i in range(4):
    row_values = [f"{V_opt_09[i*4 + j]:.3f}" for j in range(4)]
    print(f"  {row_values}") 
