import numpy as np
import gym
import random
import torch

import os

print(f"CUDA_VISIBLE_DEVICES: {os.environ.get('CUDA_VISIBLE_DEVICES', 'Not Set')}")
print(f"CUDA available: {torch.cuda.is_available()}")
print(f"PyTorch version: {torch.__version__}")
print(f"PyTorch CUDA version: {torch.version.cuda}")
if torch.cuda.is_available():
    print(f"CUDA device count: {torch.cuda.device_count()}")
    print(f"CUDA device name: {torch.cuda.get_device_name(0)}")
else:
    print("CUDA is not available. Check your installation.")


print(gym.__version__)

env = gym.make("FrozenLake-v1")
action_size = env.action_space.n
state_size = env.observation_space.n

# Use CUDA if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Create our Q table with state_size rows and action_size columns (64x4)
qtable = torch.zeros((state_size, action_size), device=device)
print(qtable)

total_episodes = 20000  # Total episodes
learning_rate = 0.7  # Learning rate
max_steps = 99  # Max steps per episode
gamma = 0.95  # Discounting rate

# Exploration parameters
epsilon = 1.0  # Exploration rate
max_epsilon = 1.0  # Exploration probability at start
min_epsilon = 0.01  # Minimum exploration probability
decay_rate = 0.005  # Exponential decay rate for exploration prob

# List of rewards
rewards = []

# Training loop
for episode in range(total_episodes):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    step = 0
    done = False
    total_rewards = 0

    for step in range(max_steps):
        exp_exp_tradeoff = random.uniform(0, 1)

        if exp_exp_tradeoff > epsilon:
            action = torch.argmax(qtable[state, :]).item()
        else:
            action = env.action_space.sample()

        # Modified to handle different return signatures
        step_result = env.step(action)
        if len(step_result) == 4:
            new_state, reward, done, info = step_result
        elif len(step_result) == 5:
            new_state, reward, done, _, info = step_result
        else:
            raise ValueError(f"Unexpected number of return values from env.step(): {len(step_result)}")

        qtable[state, action] = qtable[state, action] + learning_rate * (
            reward + gamma * torch.max(qtable[new_state, :]) - qtable[state, action]
        )

        total_rewards += reward
        state = new_state

        if done:
            break

    epsilon = min_epsilon + (max_epsilon - min_epsilon) * np.exp(-decay_rate * episode)
    rewards.append(total_rewards)

print("Score over time: " + str(sum(rewards) / total_episodes))
print(qtable)

# Test the trained agent
env = gym.make('FrozenLake-v1', render_mode='ansi')

for episode in range(5):
    state = env.reset()
    if isinstance(state, tuple):
        state = state[0]
    step = 0
    done = False
    print("****************************************************")
    print("EPISODE ", episode)

    for step in range(max_steps):
        action = torch.argmax(qtable[state, :]).item()

        # Modified to handle different return signatures
        step_result = env.step(action)
        if len(step_result) == 4:
            new_state, reward, done, info = step_result
        elif len(step_result) == 5:
            new_state, reward, done, _, info = step_result
        else:
            raise ValueError(f"Unexpected number of return values from env.step(): {len(step_result)}")

        if done:
            if new_state == 15:
                print("We reached our Goal")
            else:
                print("We fell into a hole")
            print("Number of steps", step)
            break

        state = new_state

env.close()



