# First, install the required packages
!pip install numpy gym tqdm

# Import the necessary libraries
import numpy as np
import gym
from tqdm import tqdm

# Define the epsilon-greedy policy
def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(Q[state]))
    else:
        return np.argmax(Q[state])

# Define the Q-learning algorithm
def q_learning(env, num_episodes, alpha=0.1, gamma=0.99, epsilon=0.2):  # Adjusted epsilon
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    pbar = tqdm(total=num_episodes, dynamic_ncols=True)
    for episode in range(num_episodes):
        state = env.reset()  # Adapted reset() to handle single value return
        done = False
        episode_reward = 0
        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            next_state, reward, done, _ = env.step(action)
            best_next_action = np.argmax(Q[next_state, :])
            td_target = reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            state = next_state
            episode_reward += reward
        pbar.update(1)
        if episode % 1000 == 0:
            avg_reward = evaluate_policy(env, Q, 100)
            pbar.set_description(f"\nAverage reward after {episode} episodes: {avg_reward:.2f}")
    pbar.close()
    return Q

# Evaluate the learned policy
def evaluate_policy(env, Q, num_episodes):
    total_reward = 0
    policy = np.argmax(Q, axis=1)
    for episode in range(num_episodes):
        observation = env.reset()  # Adapted reset() to handle single value return
        done = False
        episode_reward = 0
        while not done:
            action = policy[observation]
            observation, reward, done, _ = env.step(action)
            episode_reward += reward
        total_reward += episode_reward
    return total_reward / num_episodes

# Function to demo the learned agent
def demo_agent(env, Q, num_episodes=1):
    policy = np.argmax(Q, axis=1)
    for episode in range(num_episodes):
        observation = env.reset()  # Adapted reset() to handle single value return
        done = False
        print("\nEpisode:", episode + 1)
        while not done:
            env.render()
            action = policy[observation]
            observation, _, done, _ = env.step(action)
        env.render()

# Main function to run the Q-learning algorithm
def main():
    env = gym.make("FrozenLake-v1", is_slippery=True)  # Added is_slippery to introduce stochasticity
    num_episodes = 10000

    Q = q_learning(env, num_episodes)
    avg_reward = evaluate_policy(env, Q, num_episodes)
    print(f"Average reward after Q-learning: {avg_reward}")

    # Visualize the agent's performance
    visual_env = gym.make('FrozenLake-v1', render_mode='human')
    demo_agent(visual_env, Q, 3)

# Execute the main function
if __name__ == '__main__':
    main()
