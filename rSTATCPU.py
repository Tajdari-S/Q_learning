# First, install the required packages
#pip install numpy gym tqdm

tolerance = 0.1   # Tolerance for rSTAT
rho=0.1
max_epsilon = 0.99
min_epsilon = 0.01
# Import the necessary libraries
import numpy as np
import gym
from tqdm import tqdm
import random
import numpy as np
import bisect

def rSTAT(tolerance, num_samples, query_function):
    # Define a_offset
    alpha = 2 * tolerance   # Use tolerance for initial alpha
    a_offset = np.random.uniform(0, alpha)  # Random offset

    # Calculate num_regions based on the relationship provided
    num_regions = int((5 - a_offset) / alpha)  # Calculate number of regions

    # Define the regions, including the last region explicitly
    regions = [a_offset + i * alpha for i in range(num_regions)]
    regions.append(5)  # Append the endpoint of the last region
    midpoints = [(regions[i] + regions[i + 1]) / 2 for i in range(num_regions)]

    # Sample values from the query function
    sample_values = np.array([query_function() for _ in range(num_samples)]) #roh used
    avg_response = np.mean(sample_values)
    flag = 1
    if avg_response > 5:
        if avg_response < 20:
            avg_response = avg_response / 20
            flag = 20
        else:
            avg_response = avg_response / 50
            flag = 50

    # Perform binary search for the estimated region
    index = bisect.bisect_left(midpoints, avg_response)
    
    # Handle cases where avg_response is outside the range of midpoints
    estimated_region = midpoints[index]
    
    # Return the scaled estimated region
    return estimated_region * flag


# Query function to sample rewards for an action in a specific state
# Query function to sample rewards for an action in a specific state
def reward_query(state, action,env):
    transitions = env.P[state][action]
    sampled_transition = random.choices(transitions, weights=[prob for prob, _, _, _ in transitions])[0]
    prob, next_state, reward, done = sampled_transition

    #if state==14:
        #print(f"State: {state}, Action: {action}, Reward: {reward}")
    return reward

# Define the epsilon-greedy policy
def epsilon_greedy_policy(Q, state, epsilon):
    if np.random.uniform(0, 1) < epsilon:
        return np.random.choice(len(Q[state]))
    else:
        return np.argmax(Q[state])

# Define the Q-learning algorithm
def q_learning(env, num_episodes, alpha=0.6, gamma=0.95, epsilon=0.95):  # Adjusted epsilon
    Q = np.zeros([env.observation_space.n, env.action_space.n])
    pbar = tqdm(total=num_episodes, dynamic_ncols=True)
    for episode in range(num_episodes):
        epsilon = min_epsilon+(max_epsilon - min_epsilon) * np.exp(-0.0009 * episode)

        state = env.reset()  # Adapted reset() to handle single value return
        done = False
        episode_reward = 0
        while not done:
            action = epsilon_greedy_policy(Q, state, epsilon)
            estimated_reward = rSTAT(tolerance,10000, lambda: reward_query(state, action,env))

            next_state, reward, done, _ = env.step(action)


            best_next_action = np.argmax(Q[next_state, :])
            td_target = estimated_reward + gamma * Q[next_state, best_next_action]
            td_error = td_target - Q[state, action]
            Q[state, action] += alpha * td_error
            state = next_state
            episode_reward += reward
        pbar.update(1)
        #if episode % 1000 == 999:
            #avg_reward = evaluate_policy(env, Q, 100)
            #pbar.set_description(f"\nAverage reward after {episode} episodes: {avg_reward:.2f}")
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
            env_map = env.render()
            action = policy[observation]
            observation, _, done, _ = env.step(action)
            print(env_map)
        env_map =env.render()
        print(env_map)

# Main function to run the Q-learning algorithm
def main():
    env = gym.make("FrozenLake-v1", is_slippery=True,render_mode='ansi')  # Added is_slippery to introduce stochasticity
    num_episodes = 6100

    Q = q_learning(env, num_episodes)
    avg_reward = evaluate_policy(env, Q, num_episodes)
    print(f"Average reward after Q-learning: {avg_reward}")

    # Visualize the agent's performance
    #visual_env = gym.make('FrozenLake-v1', render_mode='ansi')
    #demo_agent(visual_env, Q, 3)
    return Q

# Execute the main function
if __name__ == '__main__':
    qtable=main()


print(qtable)
#display = Display(visible=0, size=(1400, 900))
#display.start()
env = gym.make('FrozenLake-v1', is_slippery=True, render_mode='ansi')  # 'ansi' is for text-based output


# Render the map and print it


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

        # Take the action (index) that have the maximum expected future reward given that state
        action = np.argmax(qtable[state,:])

        new_state, reward, done, info = env.step(action)

        if done:
            # Here, we decide to only print the last state (to see if our agent is on the goal or fall into an hole)
            #env.render()
            #print(env.render())

            if new_state == 15:
                print("We reached our Goal ")
            else:
                print("We fell into a hole ")

            # We print the number of step it took.
            print("Number of steps", step)

            break
        state = new_state
env.close()
