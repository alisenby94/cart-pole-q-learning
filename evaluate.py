"""
Evaluation script for trained Q-Learning agent on CartPole-v1.

This script loads a trained agent and evaluates its performance,
optionally rendering the environment to visualize the agent's behavior.
"""

import gymnasium as gym
import numpy as np
from agent import QLearningAgent, StateDiscretizer
import argparse
import os


def create_cartpole_discretizer():
    """
    Create state discretizer for CartPole environment.
    Must match the discretizer used during training.
    
    Returns:
        StateDiscretizer configured for CartPole
    """
    state_bounds = np.array([
        [-4.8, 4.8],      # Cart Position
        [-3.0, 3.0],      # Cart Velocity
        [-0.418, 0.418],  # Pole Angle
        [-2.0, 2.0]       # Pole Angular Velocity
    ])
    
    n_bins = np.array([10, 10, 10, 10])
    
    return StateDiscretizer(state_bounds, n_bins)


def evaluate_agent(
    model_path: str,
    n_episodes: int = 100,
    max_steps: int = 500,
    render: bool = False
):
    """
    Evaluate a trained Q-learning agent.
    
    Args:
        model_path: Path to saved agent model
        n_episodes: Number of evaluation episodes
        max_steps: Maximum steps per episode
        render: Whether to render the environment
    
    Returns:
        Dictionary with evaluation metrics
    """
    # Create environment
    if render:
        env = gym.make('CartPole-v1', render_mode='human')
    else:
        env = gym.make('CartPole-v1')
    
    # Create discretizer
    discretizer = create_cartpole_discretizer()
    
    # Load agent
    state_space_shape = tuple(discretizer.n_bins)
    n_actions = env.action_space.n
    
    agent = QLearningAgent(
        state_space_shape=state_space_shape,
        n_actions=n_actions
    )
    agent.load(model_path)
    
    print(f"Loaded agent from: {model_path}")
    print(f"Agent epsilon: {agent.epsilon:.4f}")
    print(f"Evaluating over {n_episodes} episodes...")
    print("-" * 80)
    
    # Evaluation metrics
    episode_rewards = []
    episode_lengths = []
    success_count = 0
    
    for episode in range(n_episodes):
        state, _ = env.reset()
        state_discrete = discretizer.discretize(state)
        
        total_reward = 0
        steps = 0
        
        for step in range(max_steps):
            # Select action (greedy, no exploration)
            action = agent.get_action(state_discrete, training=False)
            
            # Take action
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            next_state_discrete = discretizer.discretize(next_state)
            
            # Update state
            state_discrete = next_state_discrete
            total_reward += reward
            steps += 1
            
            if done:
                break
        
        episode_rewards.append(total_reward)
        episode_lengths.append(steps)
        
        # Count successful episodes (reward >= 195 or lasted full duration)
        if total_reward >= 195:
            success_count += 1
        
        if (episode + 1) % 10 == 0 or render:
            print(f"Episode {episode + 1}/{n_episodes} | "
                  f"Reward: {total_reward:.0f} | "
                  f"Length: {steps}")
    
    env.close()
    
    # Calculate statistics
    metrics = {
        'mean_reward': np.mean(episode_rewards),
        'std_reward': np.std(episode_rewards),
        'min_reward': np.min(episode_rewards),
        'max_reward': np.max(episode_rewards),
        'mean_length': np.mean(episode_lengths),
        'std_length': np.std(episode_lengths),
        'success_rate': success_count / n_episodes
    }
    
    # Print summary
    print("\n" + "=" * 80)
    print("EVALUATION SUMMARY")
    print("=" * 80)
    print(f"Episodes: {n_episodes}")
    print(f"Mean Reward: {metrics['mean_reward']:.2f} ± {metrics['std_reward']:.2f}")
    print(f"Min/Max Reward: {metrics['min_reward']:.0f} / {metrics['max_reward']:.0f}")
    print(f"Mean Episode Length: {metrics['mean_length']:.2f} ± {metrics['std_length']:.2f}")
    print(f"Success Rate (reward >= 195): {metrics['success_rate']:.2%}")
    print("=" * 80)
    
    return metrics


def main():
    """Main function with argument parsing."""
    parser = argparse.ArgumentParser(
        description='Evaluate a trained Q-Learning agent on CartPole-v1'
    )
    parser.add_argument(
        '--model',
        type=str,
        required=True,
        help='Path to the saved agent model (.pkl file)'
    )
    parser.add_argument(
        '--episodes',
        type=int,
        default=100,
        help='Number of evaluation episodes (default: 100)'
    )
    parser.add_argument(
        '--render',
        action='store_true',
        help='Render the environment during evaluation'
    )
    parser.add_argument(
        '--max-steps',
        type=int,
        default=500,
        help='Maximum steps per episode (default: 500)'
    )
    
    args = parser.parse_args()
    
    # Check if model file exists
    if not os.path.exists(args.model):
        print(f"Error: Model file not found: {args.model}")
        return
    
    # Evaluate agent
    evaluate_agent(
        model_path=args.model,
        n_episodes=args.episodes,
        max_steps=args.max_steps,
        render=args.render
    )


if __name__ == "__main__":
    main()
