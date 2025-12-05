import numpy as np
import pickle
from typing import Tuple, Optional


class QLearningAgent:
    def __init__(
        self,
        state_space_shape: Tuple[int, ...],
        n_actions: int,
        learning_rate: float = 0.1,
        discount_factor: float = 0.99,
        epsilon: float = 1.0,
        epsilon_decay: float = 0.995,
        epsilon_min: float = 0.01
    ):
        self.state_space_shape = state_space_shape
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min
        
        # Initialize Q-table with small random values
        self.q_table = np.random.uniform(
            low=-0.1,
            high=0.1,
            size=state_space_shape + (n_actions,)
        )
        
        # For tracking Q-table snapshots (for report)
        self.q_table_history = []
        
    def get_action(self, state: Tuple[int, ...], training: bool = True) -> int:
        if training and np.random.random() < self.epsilon:
            # Exploration: random action
            return int(np.random.randint(self.n_actions))
        else:
            # Exploitation: best known action
            return int(np.argmax(self.q_table[state]))
    
    def update(
        self,
        state: Tuple[int, ...],
        action: int,
        reward: float,
        next_state: Tuple[int, ...],
        done: bool
    ) -> None:
        current_q = self.q_table[state + (action,)]
        
        if done:
            # No future rewards if episode is done
            target_q = reward
        else:
            # Bellman equation for Q-learning
            max_next_q = np.max(self.q_table[next_state])
            target_q = reward + self.discount_factor * max_next_q
        
        # Update Q-value
        self.q_table[state + (action,)] += self.learning_rate * (target_q - current_q)
    
    def save_q_table_snapshot(self, episode: int, timestep: int, state: Tuple[int, ...], 
                              action: int, reward: float) -> None:
        snapshot = {
            'episode': episode,
            'timestep': timestep,
            'state': state,
            'action': action,
            'reward': reward,
            'q_values_for_state': self.q_table[state].copy(),
            'epsilon': self.epsilon
        }
        self.q_table_history.append(snapshot)
    
    def decay_epsilon(self) -> None:
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
    
    def save(self, filepath: str) -> None:
        data = {
            'q_table': self.q_table,
            'epsilon': self.epsilon,
            'state_space_shape': self.state_space_shape,
            'n_actions': self.n_actions,
            'learning_rate': self.learning_rate,
            'discount_factor': self.discount_factor,
            'epsilon_decay': self.epsilon_decay,
            'epsilon_min': self.epsilon_min
        }
        with open(filepath, 'wb') as f:
            pickle.dump(data, f)
    
    def load(self, filepath: str) -> None:
        with open(filepath, 'rb') as f:
            data = pickle.load(f)
        
        self.q_table = data['q_table']
        self.epsilon = data['epsilon']
        self.state_space_shape = data['state_space_shape']
        self.n_actions = data['n_actions']
        self.learning_rate = data['learning_rate']
        self.discount_factor = data['discount_factor']
        self.epsilon_decay = data['epsilon_decay']
        self.epsilon_min = data['epsilon_min']


class StateDiscretizer:
    def __init__(
        self,
        state_bounds: np.ndarray,
        n_bins: np.ndarray
    ):
        self.state_bounds = state_bounds
        self.n_bins = n_bins
        
        # Calculate bin width for each feature
        self.bin_width = (state_bounds[:, 1] - state_bounds[:, 0]) / n_bins
    
    def discretize(self, state: np.ndarray) -> Tuple[int, ...]:
        # Clip state to bounds
        state_clipped = np.clip(state, self.state_bounds[:, 0], self.state_bounds[:, 1])
        
        # Calculate bin indices
        state_normalized = (state_clipped - self.state_bounds[:, 0]) / self.bin_width
        state_discrete = np.floor(state_normalized).astype(int)
        
        # Ensure within bounds (handle edge case where state equals upper bound)
        state_discrete = np.clip(state_discrete, 0, self.n_bins - 1)
        
        return tuple(state_discrete)
