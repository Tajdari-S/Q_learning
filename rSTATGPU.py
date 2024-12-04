# First, install the required packages
#pip install numpy gym tqdm torch

tolerance = 0.1   # Tolerance for rSTAT
rho=0.1
max_epsilon = 0.99
min_epsilon = 0.01
# Import the necessary libraries
import numpy as np
import gym
from tqdm import tqdm
import random
import bisect
import torch
import torch.cuda

# Check if CUDA is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")

def rSTAT(tolerance, num_samples, query_function):
    # Keep random operations on CPU
    alpha = 2 * tolerance
    a_offset = np.random.uniform(0, alpha)
    num_regions = int((5 - a_offset) / alpha)
    
    regions = [a_offset + i * alpha for i in range(num_regions)]
    regions.append(5)
    midpoints = [(regions[i] + regions[i + 1]) / 2 for i in range(num_regions)]
    
    # Sample values on CPU
    sample_values = np.array([query_function() for _ in range(num_samples)])
    
    # Transfer to GPU for computation
    sample_tensor = torch.tensor(sample_values, device=device)
    avg_response = torch.mean(sample_tensor).item()
    
    flag = 1
    if avg_response > 5:
        if avg_response < 20:
            avg_response = avg_response / 20
            flag = 20
        else:
            avg_response = avg_response / 50
            flag = 50

    index = bisect.bisect_left(midpoints, avg_response)
    estimated_region = midpoints[index]
    return estimated_region * flag

def reward_query(state, action, env):
    # Keep environment interactions on CPU
    transitions = env.P[state][action]
    sampled_transition = random.choices(transitions, weights=[prob for prob, _, _, _ in transitions])[0]
    prob, next_state, reward, done = sampled_transition
    return reward

def epsilon_greedy_policy(Q, state, epsilon):
    # Random choice on CPU, max operation on GPU
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(Q.shape[1])
    else:
        with torch.no_grad():
            state_actions = Q[state]
            return torch.argmax(state_actions).item()

def q_learning(env, num_episodes, alpha=0.6, gamma=0.95, epsilon=0.95):
    # Initialize Q-table on GPU
    Q = torch.zeros((env.observation_space.n, env.action_space.n), device=device)
    
    # Create buffer for batch updates
    batch_size = 100
    update_buffer = torch.zeros((batch_size, 3), device=device)  # [state, action, update]
    buffer_idx = 0
    
    pbar = tqdm(total=num_episodes, dynamic_ncols=True)
    for episode in range(num_episodes):
        epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-0.0009 * episode)
        state = env.reset()
        done = False
        
        while not done:
            # CPU operations for policy and environment
            action = epsilon_greedy_policy(Q, state, epsilon)
            estimated_reward = rSTAT(tolerance, 10000, lambda: reward_query(state, action, env))
            next_state, reward, done, _ = env.step(action)
            
            # GPU operations for Q-value updates
            with torch.no_grad():
                best_next_action = torch.argmax(Q[next_state]).item()
                td_target = estimated_reward + gamma * Q[next_state, best_next_action]
                td_error = td_target - Q[state, action]
                
                # Add to update buffer
                update_buffer[buffer_idx] = torch.tensor([state, action, alpha * td_error], 
                                                       device=device)
                buffer_idx += 1
                
                # Apply updates when buffer is full
                if buffer_idx == batch_size:
                    states = update_buffer[:, 0].long()
                    actions = update_buffer[:, 1].long()
                    updates = update_buffer[:, 2]
                    
                    # Batch update Q-values
                    Q[states, actions] += updates
                    
                    # Reset buffer
                    buffer_idx = 0
                    update_buffer.zero_()
            
            state = next_state
            
        pbar.update(1)
    
    # Apply remaining updates
    if buffer_idx > 0:
        states = update_buffer[:buffer_idx, 0].long()
        actions = update_buffer[:buffer_idx, 1].long()
        updates = update_buffer[:buffer_idx, 2]
        Q[states, actions] += updates
    
    pbar.close()
    return Q

def evaluate_policy(env, Q, num_episodes):
    total_reward = 0
    # Move Q-table to CPU for policy extraction
    Q_cpu = Q.cpu()
    policy = torch.argmax(Q_cpu, dim=1).numpy()
    
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        episode_reward = 0
        while not done:
            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / num_episodes

def demo_agent(env, Q, num_episodes=1):
    # Move Q-table to CPU for visualization
    Q_cpu = Q.cpu()
    policy = torch.argmax(Q_cpu, dim=1).numpy()
    
    for episode in range(num_episodes):
        observation = env.reset()
        done = False
        print("\nEpisode:", episode + 1)
        while not done:
            env_map = env.render()
            action = policy[observation]
            observation, _, done, _ = env.step(action)
            print(env_map)
        env_map = env.render()
        print(env_map)

def main():
    env = gym.make("FrozenLake-v1", is_slippery=True, render_mode='ansi')
    num_episodes = 7100

    Q = q_learning(env, num_episodes)
    avg_reward = evaluate_policy(env, Q, num_episodes)
    print(f"Average reward after Q-learning: {avg_reward}")
    return Q

if __name__ == '__main__':
    qtable = main()
    # Move Q-table to CPU for final evaluation
    qtable_cpu = qtable.cpu().numpy()
    
    env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='ansi')
    max_steps = 100

    for episode in range(5):
        state = env.reset()
        if episode == 0:
            env_map = env.render()
            print(env_map)
        step = 0
        done = False
        print("****************************************************")
        print("EPISODE ", episode)

        for step in range(max_steps):
            action = np.argmax(qtable_cpu[state,:])
            new_state, reward, done, info = env.step(action)

            if done:
                if new_state == 15:
                    print("We reached our Goal ")
                else:
                    print("We fell into a hole ")
                print("Number of steps", step)
                break
            state = new_state
    env.close()
