"""
Reinforcement Learning Algorithms

Implements:
- Q-Learning
- Policy Gradient
- Deep Q-Network (DQN)
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Callable
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class QLearning:
    """
    Q-Learning Algorithm
    
    Model-free reinforcement learning
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.1,
                 discount_factor: float = 0.9, epsilon: float = 0.1):
        """
        Initialize Q-Learning
        
        Parameters
        ----------
        n_states : int
            Number of states
        n_actions : int
            Number of actions
        learning_rate : float
            Learning rate (alpha)
        discount_factor : float
            Discount factor (gamma)
        epsilon : float
            Epsilon for epsilon-greedy policy
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        
        # Q-table
        self.Q = np.zeros((n_states, n_actions))
    
    def get_action(self, state: int, training: bool = True) -> int:
        """
        Get action using epsilon-greedy policy
        
        Parameters
        ----------
        state : int
            Current state
        training : bool
            Whether in training mode
            
        Returns
        -------
        action : int
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            # Explore: random action
            return np.random.randint(self.n_actions)
        else:
            # Exploit: best action
            return np.argmax(self.Q[state])
    
    def update(self, state: int, action: int, reward: float, next_state: int, done: bool):
        """
        Update Q-value using Q-learning update rule
        
        Parameters
        ----------
        state : int
            Current state
        action : int
            Action taken
        reward : float
            Reward received
        next_state : int
            Next state
        done : bool
            Whether episode is done
        """
        if done:
            # Terminal state
            target = reward
        else:
            # Q-learning update: Q(s,a) = Q(s,a) + alpha * (r + gamma * max Q(s',a') - Q(s,a))
            target = reward + self.discount_factor * np.max(self.Q[next_state])
        
        # Update Q-value
        self.Q[state, action] += self.learning_rate * (target - self.Q[state, action])
    
    def get_policy(self) -> np.ndarray:
        """
        Get greedy policy from Q-table
        
        Returns
        -------
        policy : array
            Policy (state -> action)
        """
        return np.argmax(self.Q, axis=1)


class PolicyGradient:
    """
    Policy Gradient Method
    
    REINFORCE algorithm
    """
    
    def __init__(self, n_states: int, n_actions: int, learning_rate: float = 0.01):
        """
        Initialize policy gradient
        
        Parameters
        ----------
        n_states : int
            Number of states
        n_actions : int
            Number of actions
        learning_rate : float
            Learning rate
        """
        self.n_states = n_states
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        
        # Policy parameters (state -> action probabilities)
        self.theta = np.random.randn(n_states, n_actions) * 0.01
    
    def get_policy(self, state: int) -> np.ndarray:
        """
        Get action probabilities for state
        
        Parameters
        ----------
        state : int
            Current state
            
        Returns
        -------
        probabilities : array
            Action probabilities
        """
        # Softmax
        exp_theta = np.exp(self.theta[state] - np.max(self.theta[state]))
        return exp_theta / np.sum(exp_theta)
    
    def get_action(self, state: int) -> int:
        """
        Sample action from policy
        
        Parameters
        ----------
        state : int
            Current state
            
        Returns
        -------
        action : int
            Sampled action
        """
        probs = self.get_policy(state)
        return np.random.choice(self.n_actions, p=probs)
    
    def update(self, episode: List[Tuple[int, int, float]]):
        """
        Update policy using REINFORCE
        
        Parameters
        ----------
        episode : list of (state, action, reward) tuples
            Episode trajectory
        """
        # Calculate returns (discounted rewards)
        returns = []
        G = 0
        for _, _, reward in reversed(episode):
            G = reward + 0.9 * G  # Discount factor = 0.9
            returns.insert(0, G)
        
        # Update policy
        for (state, action, _), return_val in zip(episode, returns):
            probs = self.get_policy(state)
            
            # Policy gradient
            grad = np.zeros(self.n_actions)
            grad[action] = 1.0 / (probs[action] + 1e-10)
            grad = grad - probs
            
            # Update parameters
            self.theta[state] += self.learning_rate * return_val * grad


class DQN:
    """
    Deep Q-Network (DQN)
    
    Q-Learning with neural network function approximation
    """
    
    def __init__(self, state_dim: int, n_actions: int, learning_rate: float = 0.001,
                 discount_factor: float = 0.99, epsilon: float = 0.1, epsilon_decay: float = 0.995):
        """
        Initialize DQN
        
        Parameters
        ----------
        state_dim : int
            State dimension
        n_actions : int
            Number of actions
        learning_rate : float
            Learning rate
        discount_factor : float
            Discount factor
        epsilon : float
            Initial epsilon
        epsilon_decay : float
            Epsilon decay rate
        """
        self.state_dim = state_dim
        self.n_actions = n_actions
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay
        
        # Simple neural network (2-layer MLP)
        self.W1 = np.random.randn(state_dim, 64) * 0.01
        self.b1 = np.zeros(64)
        self.W2 = np.random.randn(64, n_actions) * 0.01
        self.b2 = np.zeros(n_actions)
        
        # Experience replay buffer
        self.replay_buffer = []
        self.buffer_size = 10000
        self.batch_size = 32
    
    def _forward(self, state: np.ndarray) -> np.ndarray:
        """Forward pass through network"""
        # Layer 1
        h1 = np.maximum(0, state @ self.W1 + self.b1)  # ReLU
        # Layer 2
        q_values = h1 @ self.W2 + self.b2
        return q_values
    
    def get_action(self, state: np.ndarray, training: bool = True) -> int:
        """
        Get action using epsilon-greedy policy
        
        Parameters
        ----------
        state : array
            Current state
        training : bool
            Whether in training mode
            
        Returns
        -------
        action : int
            Selected action
        """
        if training and np.random.random() < self.epsilon:
            return np.random.randint(self.n_actions)
        else:
            q_values = self._forward(state)
            return np.argmax(q_values)
    
    def store_transition(self, state: np.ndarray, action: int, reward: float,
                        next_state: np.ndarray, done: bool):
        """Store transition in replay buffer"""
        self.replay_buffer.append((state, action, reward, next_state, done))
        if len(self.replay_buffer) > self.buffer_size:
            self.replay_buffer.pop(0)
    
    def train_step(self):
        """Train on batch from replay buffer"""
        if len(self.replay_buffer) < self.batch_size:
            return
        
        # Sample batch
        batch = np.random.choice(len(self.replay_buffer), self.batch_size, replace=False)
        batch_transitions = [self.replay_buffer[i] for i in batch]
        
        states = np.array([t[0] for t in batch_transitions])
        actions = np.array([t[1] for t in batch_transitions])
        rewards = np.array([t[2] for t in batch_transitions])
        next_states = np.array([t[3] for t in batch_transitions])
        dones = np.array([t[4] for t in batch_transitions])
        
        # Current Q-values
        current_q = self._forward(states)
        
        # Target Q-values
        next_q = self._forward(next_states)
        target_q = current_q.copy()
        
        for i in range(self.batch_size):
            if dones[i]:
                target_q[i, actions[i]] = rewards[i]
            else:
                target_q[i, actions[i]] = rewards[i] + self.discount_factor * np.max(next_q[i])
        
        # Gradient descent (simplified)
        error = target_q - current_q
        grad = error / self.batch_size
        
        # Update network (simplified backpropagation)
        # In practice, use automatic differentiation
        self.W2 += self.learning_rate * np.mean(grad[:, :, np.newaxis] * 
                                                np.maximum(0, states @ self.W1 + self.b1)[:, :, np.newaxis], axis=0)
        
        # Decay epsilon
        self.epsilon = max(0.01, self.epsilon * self.epsilon_decay)
