import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
from agent import QLearningAgent, StateDiscretizer
import os
from datetime import datetime


def create_cartpole_discretizer():
    # Define state bounds
    state_bounds = np.array([
        [-4.8, 4.8],      # Cart Position
        [-3.0, 3.0],      # Cart Velocity
        [-0.418, 0.418],  # Pole Angle
        [-2.0, 2.0]       # Pole Angular Velocity
    ])
    
    # Number of bins for each state feature
    n_bins = np.array([10, 10, 10, 10])
    
    return StateDiscretizer(state_bounds, n_bins)


def train_agent(
    n_episodes: int = 10000,
    max_steps: int = 500,
    learning_rate: float = 0.1,
    discount_factor: float = 0.99,
    epsilon: float = 1.0,
    epsilon_decay: float = 0.995,
    epsilon_min: float = 0.01,
    render: bool = False,
    save_path: str = 'models'
):
    # Create environment
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    
    # Create state discretizer
    discretizer = create_cartpole_discretizer()
    
    # Create agent
    state_space_shape = tuple(discretizer.n_bins)
    n_actions = env.action_space.n
    
    agent = QLearningAgent(
        state_space_shape=state_space_shape,
        n_actions=n_actions,
        learning_rate=learning_rate,
        discount_factor=discount_factor,
        epsilon=epsilon,
        epsilon_decay=epsilon_decay,
        epsilon_min=epsilon_min
    )
    
    # Training metrics
    episode_rewards = []
    episode_lengths = []
    
    print("Starting training...")
    print(f"Episodes: {n_episodes}, Max Steps: {max_steps}")
    print(f"Learning Rate: {learning_rate}, Discount Factor: {discount_factor}")
    print(f"Initial Epsilon: {epsilon}, Epsilon Decay: {epsilon_decay}, Min Epsilon: {epsilon_min}")
    print("-" * 80)
    
    last_episode_data = []  # Store data for last episode to get final 5 timesteps
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        state_discrete = discretizer.discretize(state)
        
        total_reward = 0
        steps = 0
        episode_data = []  # Store all timesteps for this episode
        
        for step in range(max_steps):
            # Select action
            action = agent.get_action(state_discrete, training=True)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_discrete = discretizer.discretize(next_state)
            
            # Store episode data (we'll save last episode's last 5 timesteps at the end)
            episode_data.append({
                'step': step,
                'state': state_discrete,
                'action': action,
                'reward': reward
            })
            
            # Save Q-table snapshot for first 5 timesteps of first episode
            if episode == 0 and step < 5:
                agent.save_q_table_snapshot(episode, step, state_discrete, action, reward)
            
            # Update agent
            agent.update(state_discrete, action, reward, next_state_discrete, done)
            
            # Update state
            state_discrete = next_state_discrete
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        # Store this episode's data as potential last episode
        last_episode_data = episode_data
        
        # Decay epsilon
        agent.decay_epsilon()
        
        # Record metrics
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Print progress
        if (episode + 1) % 100 == 0:
            avg_reward = np.mean(episode_rewards[-100:])
            avg_length = np.mean(episode_lengths[-100:])
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Avg Reward (last 100): {avg_reward:.2f} | "
                  f"Avg Length (last 100): {avg_length:.2f} | "
                  f"Epsilon: {agent.epsilon:.4f}")
        
        # Check if solved (average reward of 195 over 100 consecutive episodes)
        if len(episode_rewards) >= 100:
            avg_reward_100 = np.mean(episode_rewards[-100:])
            if avg_reward_100 >= 195.0:
                print(f"\nEnvironment solved in {episode + 1} episodes!")
                print(f"Average reward over last 100 episodes: {avg_reward_100:.2f}")
                break
    
    env.close()
    
    # Save Q-table snapshots for last 5 timesteps of final episode
    if len(last_episode_data) >= 5:
        final_episode = len(episode_rewards) - 1
        for i in range(max(0, len(last_episode_data) - 5), len(last_episode_data)):
            data = last_episode_data[i]
            agent.save_q_table_snapshot(
                final_episode, 
                data['step'], 
                data['state'], 
                data['action'], 
                data['reward']
            )
    
    # Create save directory if it doesn't exist
    os.makedirs(save_path, exist_ok=True)
    
    # Save agent
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_path = os.path.join(save_path, f'cartpole_agent_{timestamp}.pkl')
    agent.save(model_path)
    print(f"\nModel saved to: {model_path}")
    
    # Plot training progress
    plot_training_progress(episode_rewards, episode_lengths, save_path, timestamp)
    
    # Save Q-table snapshots to file for report
    save_q_table_snapshots(agent.q_table_history, save_path, timestamp)
    
    return agent, episode_rewards, episode_lengths


def save_q_table_snapshots(q_table_history, save_path, timestamp):
    snapshot_path = os.path.join(save_path, f'q_table_snapshots_{timestamp}.txt')
    
    with open(snapshot_path, 'w') as f:
        f.write("=" * 80 + "\n")
        f.write("Q-TABLE SNAPSHOTS FOR REPORT\n")
        f.write("=" * 80 + "\n\n")
        
        # First 5 timesteps of episode 1
        f.write("FIRST 5 TIMESTEPS OF EPISODE 1:\n")
        f.write("-" * 80 + "\n")
        first_episode_snapshots = [s for s in q_table_history if s['episode'] == 0]
        for snapshot in first_episode_snapshots[:5]:
            f.write(f"\nTimestep {snapshot['timestep']}:\n")
            f.write(f"  State (discretized): {snapshot['state']}\n")
            f.write(f"  Action taken: {snapshot['action']}\n")
            f.write(f"  Reward: {snapshot['reward']}\n")
            f.write(f"  Q-values for this state: {snapshot['q_values_for_state']}\n")
            f.write(f"  Epsilon: {snapshot['epsilon']:.4f}\n")
        
        # Last 5 timesteps of last episode
        f.write("\n" + "=" * 80 + "\n")
        f.write("LAST 5 TIMESTEPS OF FINAL EPISODE:\n")
        f.write("-" * 80 + "\n")
        last_episode_num = max([s['episode'] for s in q_table_history])
        last_episode_snapshots = [s for s in q_table_history if s['episode'] == last_episode_num]
        for snapshot in last_episode_snapshots[-5:]:
            f.write(f"\nTimestep {snapshot['timestep']}:\n")
            f.write(f"  State (discretized): {snapshot['state']}\n")
            f.write(f"  Action taken: {snapshot['action']}\n")
            f.write(f"  Reward: {snapshot['reward']}\n")
            f.write(f"  Q-values for this state: {snapshot['q_values_for_state']}\n")
            f.write(f"  Epsilon: {snapshot['epsilon']:.4f}\n")
        
        f.write("\n" + "=" * 80 + "\n")
    
    print(f"Q-table snapshots saved to: {snapshot_path}")
    
    # Also save as JSON for easier programmatic access
    import json
    json_path = os.path.join(save_path, f'q_table_snapshots_{timestamp}.json')
    # Convert numpy arrays and types to native Python types for JSON serialization
    json_data = []
    for snapshot in q_table_history:
        json_snapshot = {
            'episode': int(snapshot['episode']),
            'timestep': int(snapshot['timestep']),
            'state': tuple(int(x) for x in snapshot['state']),
            'action': int(snapshot['action']),
            'reward': float(snapshot['reward']),
            'q_values_for_state': snapshot['q_values_for_state'].tolist(),
            'epsilon': float(snapshot['epsilon'])
        }
        json_data.append(json_snapshot)
    
    with open(json_path, 'w') as f:
        json.dump(json_data, f, indent=2)
    print(f"Q-table snapshots (JSON) saved to: {json_path}")


def plot_training_progress(rewards, lengths, save_path, timestamp):
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
    
    # Plot rewards
    ax1.plot(rewards, alpha=0.6, color='blue', linewidth=0.5)
    if len(rewards) >= 100:
        # Plot moving average
        moving_avg = np.convolve(rewards, np.ones(100)/100, mode='valid')
        ax1.plot(range(99, len(rewards)), moving_avg, color='red', linewidth=2, label='100-episode moving average')
        ax1.axhline(y=195, color='green', linestyle='--', label='Solved threshold (195)')
    ax1.set_xlabel('Episode')
    ax1.set_ylabel('Total Reward')
    ax1.set_title('Training Progress: Episode Rewards')
    ax1.legend()
    ax1.grid(True, alpha=0.3)
    
    # Plot episode lengths
    ax2.plot(lengths, alpha=0.6, color='blue', linewidth=0.5)
    if len(lengths) >= 100:
        moving_avg_length = np.convolve(lengths, np.ones(100)/100, mode='valid')
        ax2.plot(range(99, len(lengths)), moving_avg_length, color='red', linewidth=2, label='100-episode moving average')
    ax2.set_xlabel('Episode')
    ax2.set_ylabel('Episode Length')
    ax2.set_title('Training Progress: Episode Lengths')
    ax2.legend()
    ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(save_path, f'training_progress_{timestamp}.png')
    plt.savefig(plot_path, dpi=150)
    print(f"Training plot saved to: {plot_path}")
    plt.close()


if __name__ == "__main__":
    # Train agent with default parameters
    agent, rewards, lengths = train_agent(
        n_episodes=10000,
        max_steps=500,
        learning_rate=0.1,
        discount_factor=0.99,
        epsilon=1.0,
        epsilon_decay=0.995,
        epsilon_min=0.01,
        render=False,
        save_path='models'
    )
    
    print("\nTraining completed!")
    print(f"Total episodes: {len(rewards)}")
    print(f"Final average reward (last 100 episodes): {np.mean(rewards[-100:]):.2f}")
    print(f"Best episode reward: {max(rewards):.2f}")
