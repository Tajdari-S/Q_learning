!pip install gymnasium numpy gym tqdm
import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from IPython.display import display, Image
import imageio.v2 as imageio
from tqdm import tqdm
import os

class FrozenLakeAgent:
    def __init__(self, env, learning_rate=0.1, discount_factor=0.9, epsilon=0.1):
        self.env = env
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.q_table = np.zeros([env.observation_space.n, env.action_space.n])
        
    def choose_action(self, state):
        if np.random.random() < self.epsilon:
            return self.env.action_space.sample()
        return np.argmax(self.q_table[state])

    def learn(self, state, action, reward, next_state, done):
        # Modified reward structure
        if done and reward == 0:  # If fell in hole
            reward = -1
        elif done and reward == 1:  # If reached goal
            reward = 10
        elif not done:  # Small negative reward for each step to encourage shorter paths
            reward = -0.01
            
        old_value = self.q_table[state, action]
        if done:
            next_max = 0
        else:
            next_max = np.max(self.q_table[next_state])
        new_value = reward + self.discount_factor * next_max
        self.q_table[state, action] = (1 - self.learning_rate) * old_value + self.learning_rate * new_value

def train_agent(env, agent, episodes=2000):
    rewards_history = []
    success_history = []
    frames_buffer = []

    # Ensure the gifs directory exists
    if not os.path.exists('gifs'):
        os.makedirs('gifs')

    # Training loop
    for episode in tqdm(range(episodes), desc="Training"):
        state = env.reset()[0]
        frames = []
        total_reward = 0
        done = False

        # Episode loop
        while not done:
            frame = env.render()
            frames.append(frame)

            action = agent.choose_action(state)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # Modified to pass done status to learn method
            agent.learn(state, action, reward, next_state, done)

            total_reward += reward
            state = next_state

        rewards_history.append(total_reward)
        success_history.append(1 if total_reward > 0 else 0)

        if total_reward > 0 or episode % 500 == 0:
            save_frames_as_gif(frames, 'gifs', f'episode_{episode}_reward_{total_reward}.gif')
            frames_buffer = frames.copy()

        if episode % 100 == 0:
            success_rate = np.mean(success_history[-100:]) if success_history else 0
            print(f"\nEpisode {episode}")
            print(f"Success rate: {success_rate:.2%}")
            print(f"Average reward: {np.mean(rewards_history[-100:]):.3f}")

    return rewards_history, success_history, frames_buffer

def plot_training_progress(rewards, successes):
    plt.figure(figsize=(12, 4))

    # Plot rewards
    plt.subplot(1, 2, 1)
    plt.plot(rewards, alpha=0.6)
    plt.plot(np.convolve(rewards, np.ones(100)/100, mode='valid'),
             label='Moving Average (100 episodes)')
    plt.title('Rewards over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Total Reward')
    plt.legend()

    # Plot success rate
    plt.subplot(1, 2, 2)
    window = 100
    success_rate = [np.mean(successes[max(0, i-window):i+1])
                   for i in range(len(successes))]
    plt.plot(success_rate)
    plt.title('Success Rate over Episodes')
    plt.xlabel('Episode')
    plt.ylabel('Success Rate')

    plt.tight_layout()
    plt.show()

def main():
    # Create environment with rgb_array render mode
    env = gym.make('FrozenLake-v1', render_mode='rgb_array', is_slippery=False)

    # Initialize agent with adjusted parameters
    agent = FrozenLakeAgent(
        env=env,
        learning_rate=0.1,
        discount_factor=0.9,  # Slightly reduced to prioritize shorter paths
        epsilon=0.2  # Increased exploration
    )

    # Train agent
    print("Starting training...")
    rewards_history, success_history, final_frames = train_agent(
        env=env,
        agent=agent,
        episodes=200  # Increased number of episodes
    )

    # Display final statistics
    final_success_rate = np.mean(success_history[-100:])
    final_avg_reward = np.mean(rewards_history[-100:])

    print("\nTraining Complete!")
    print(f"Final Success Rate: {final_success_rate:.2%}")
    print(f"Final Average Reward: {final_avg_reward:.3f}")

    # Display some successful episode GIFs
    print("\nDisplaying some episode GIFs:")
    gif_files = sorted([f for f in os.listdir('gifs') if f.endswith('.gif')])
    for gif_file in gif_files[-3:]:  # Show last 3 GIFs
        print(f"\nShowing {gif_file}")
        display(Image(filename=os.path.join('gifs', gif_file)))

    env.close()

if __name__ == "__main__":
    main()
