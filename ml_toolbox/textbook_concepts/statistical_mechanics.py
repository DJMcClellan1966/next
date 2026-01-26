"""
Statistical Mechanics & Thermodynamics - Inspired by Ludwig Boltzmann

Implements:
- Simulated Annealing
- Boltzmann Machines
- Temperature Scheduling
- Entropy Regularization
- Free Energy Optimization
"""
import numpy as np
from typing import Callable, Dict, Any, Optional, List, Tuple
import random
import logging
from scipy.special import expit  # Sigmoid function

logger = logging.getLogger(__name__)


class SimulatedAnnealing:
    """
    Simulated Annealing - Global optimization inspired by metal cooling
    
    Uses temperature to control exploration vs exploitation
    """
    
    def __init__(
        self,
        objective_function: Callable,
        initial_solution: np.ndarray,
        bounds: Optional[List[Tuple[float, float]]] = None,
        initial_temperature: float = 100.0,
        cooling_rate: float = 0.95,
        min_temperature: float = 0.01,
        max_iterations: int = 1000,
        step_size: float = 1.0
    ):
        """
        Initialize Simulated Annealing
        
        Args:
            objective_function: Function to minimize
            initial_solution: Starting point
            bounds: Optional bounds for each dimension
            initial_temperature: Starting temperature
            cooling_rate: Temperature decay rate (0 < rate < 1)
            min_temperature: Minimum temperature (stopping criterion)
            max_iterations: Maximum iterations
            step_size: Step size for random walk
        """
        self.objective_function = objective_function
        self.current_solution = np.array(initial_solution).copy()
        self.bounds = bounds
        self.temperature = initial_temperature
        self.initial_temperature = initial_temperature
        self.cooling_rate = cooling_rate
        self.min_temperature = min_temperature
        self.max_iterations = max_iterations
        self.step_size = step_size
        
        self.best_solution = self.current_solution.copy()
        self.best_energy = self.objective_function(self.current_solution)
        self.current_energy = self.best_energy
        self.energy_history = [self.best_energy]
        self.temperature_history = [self.temperature]
    
    def _generate_neighbor(self, solution: np.ndarray) -> np.ndarray:
        """Generate neighboring solution via random walk"""
        neighbor = solution + np.random.normal(0, self.step_size, size=solution.shape)
        
        # Apply bounds if provided
        if self.bounds:
            for i, (min_val, max_val) in enumerate(self.bounds):
                neighbor[i] = np.clip(neighbor[i], min_val, max_val)
        
        return neighbor
    
    def _acceptance_probability(self, new_energy: float, current_energy: float) -> float:
        """Calculate acceptance probability (Metropolis criterion)"""
        if new_energy < current_energy:
            return 1.0
        return np.exp(-(new_energy - current_energy) / self.temperature)
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run simulated annealing
        
        Returns:
            Dictionary with best solution, energy, and history
        """
        iteration = 0
        
        while self.temperature > self.min_temperature and iteration < self.max_iterations:
            # Generate neighbor
            neighbor = self._generate_neighbor(self.current_solution)
            neighbor_energy = self.objective_function(neighbor)
            
            # Accept or reject
            if random.random() < self._acceptance_probability(neighbor_energy, self.current_energy):
                self.current_solution = neighbor
                self.current_energy = neighbor_energy
                
                # Update best
                if neighbor_energy < self.best_energy:
                    self.best_solution = neighbor.copy()
                    self.best_energy = neighbor_energy
            
            # Cool down
            self.temperature *= self.cooling_rate
            iteration += 1
            
            # Track history
            self.energy_history.append(self.best_energy)
            self.temperature_history.append(self.temperature)
        
        return {
            'best_solution': self.best_solution,
            'best_energy': self.best_energy,
            'energy_history': self.energy_history,
            'temperature_history': self.temperature_history,
            'iterations': iteration
        }


class BoltzmannMachine:
    """
    Boltzmann Machine - Energy-based model for unsupervised learning
    
    Uses energy function and Boltzmann distribution
    """
    
    def __init__(
        self,
        n_visible: int,
        n_hidden: int = 0,
        learning_rate: float = 0.1,
        temperature: float = 1.0
    ):
        """
        Initialize Boltzmann Machine
        
        Args:
            n_visible: Number of visible units
            n_hidden: Number of hidden units (0 for restricted BM)
            learning_rate: Learning rate
            temperature: Temperature parameter
        """
        self.n_visible = n_visible
        self.n_hidden = n_hidden
        self.n_units = n_visible + n_hidden
        self.learning_rate = learning_rate
        self.temperature = temperature
        
        # Initialize weights and biases
        self.weights = np.random.normal(0, 0.1, (self.n_units, self.n_units))
        self.weights = (self.weights + self.weights.T) / 2  # Symmetric
        np.fill_diagonal(self.weights, 0)  # No self-connections
        
        self.biases = np.random.normal(0, 0.1, self.n_units)
    
    def _energy(self, state: np.ndarray) -> float:
        """Calculate energy of state"""
        return -0.5 * np.dot(state, np.dot(self.weights, state)) - np.dot(self.biases, state)
    
    def _sigmoid(self, x: float) -> float:
        """Sigmoid activation"""
        return expit(x / self.temperature)
    
    def _sample_hidden(self, visible: np.ndarray) -> np.ndarray:
        """Sample hidden units given visible"""
        if self.n_hidden == 0:
            return np.array([])
        
        hidden_inputs = np.dot(self.weights[:self.n_visible, self.n_visible:], visible) + \
                       self.biases[self.n_visible:]
        hidden_probs = self._sigmoid(hidden_inputs)
        return (np.random.random(self.n_hidden) < hidden_probs).astype(float)
    
    def _sample_visible(self, hidden: np.ndarray) -> np.ndarray:
        """Sample visible units given hidden"""
        visible_inputs = np.dot(self.weights[self.n_visible:, :self.n_visible], hidden) + \
                        self.biases[:self.n_visible]
        visible_probs = self._sigmoid(visible_inputs)
        return (np.random.random(self.n_visible) < visible_probs).astype(float)
    
    def _gibbs_sampling(self, visible: np.ndarray, n_steps: int = 10) -> Tuple[np.ndarray, np.ndarray]:
        """Gibbs sampling for contrastive divergence"""
        current_visible = visible.copy()
        
        for _ in range(n_steps):
            hidden = self._sample_hidden(current_visible)
            if len(hidden) > 0:
                current_visible = self._sample_visible(hidden)
            else:
                # Restricted BM - just sample visible
                visible_inputs = np.dot(self.weights[:self.n_visible, :self.n_visible], current_visible) + \
                               self.biases[:self.n_visible]
                visible_probs = self._sigmoid(visible_inputs)
                current_visible = (np.random.random(self.n_visible) < visible_probs).astype(float)
        
        reconstructed_hidden = self._sample_hidden(current_visible)
        return current_visible, reconstructed_hidden
    
    def fit(self, X: np.ndarray, epochs: int = 100, batch_size: int = 32) -> Dict[str, Any]:
        """
        Train Boltzmann Machine using contrastive divergence
        
        Args:
            X: Training data (binary or normalized to [0,1])
            epochs: Number of training epochs
            batch_size: Batch size
        
        Returns:
            Training history
        """
        n_samples = X.shape[0]
        history = {'energy': [], 'reconstruction_error': []}
        
        for epoch in range(epochs):
            epoch_energy = 0
            epoch_recon_error = 0
            
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for i in range(0, n_samples, batch_size):
                batch_indices = indices[i:i + batch_size]
                batch = X[batch_indices]
                
                # Positive phase: data distribution
                positive_hidden = np.array([self._sample_hidden(x) for x in batch])
                if self.n_hidden == 0:
                    positive_visible = batch
                    positive_hidden = np.zeros((len(batch), 0))
                else:
                    positive_visible = batch
                
                # Negative phase: model distribution (Gibbs sampling)
                negative_visible, negative_hidden = zip(*[self._gibbs_sampling(x) for x in batch])
                negative_visible = np.array(negative_visible)
                if self.n_hidden > 0:
                    negative_hidden = np.array(negative_hidden)
                else:
                    negative_hidden = np.zeros((len(batch), 0))
                
                # Update weights (contrastive divergence)
                if self.n_hidden > 0:
                    positive_gradient = np.outer(positive_visible.mean(axis=0), 
                                                 positive_hidden.mean(axis=0))
                    negative_gradient = np.outer(negative_visible.mean(axis=0),
                                                 negative_hidden.mean(axis=0))
                else:
                    positive_gradient = np.outer(positive_visible.mean(axis=0),
                                                 positive_visible.mean(axis=0))
                    negative_gradient = np.outer(negative_visible.mean(axis=0),
                                                 negative_visible.mean(axis=0))
                
                weight_gradient = positive_gradient - negative_gradient
                
                # Update weights (only upper triangle, then symmetrize)
                self.weights[:self.n_visible, self.n_visible:] += \
                    self.learning_rate * weight_gradient
                if self.n_hidden > 0:
                    self.weights[self.n_visible:, :self.n_visible] = \
                        self.weights[:self.n_visible, self.n_visible:].T
                
                # Update biases
                self.biases[:self.n_visible] += self.learning_rate * \
                    (positive_visible.mean(axis=0) - negative_visible.mean(axis=0))
                if self.n_hidden > 0:
                    self.biases[self.n_visible:] += self.learning_rate * \
                        (positive_hidden.mean(axis=0) - negative_hidden.mean(axis=0))
                
                # Calculate energy and reconstruction error
                epoch_energy += np.mean([self._energy(x) for x in batch])
                epoch_recon_error += np.mean(np.abs(batch - negative_visible))
            
            history['energy'].append(epoch_energy / (n_samples / batch_size))
            history['reconstruction_error'].append(epoch_recon_error / (n_samples / batch_size))
        
        return history
    
    def sample(self, n_samples: int = 1, n_gibbs_steps: int = 100) -> np.ndarray:
        """Generate samples from the model"""
        samples = []
        initial_visible = np.random.random((1, self.n_visible))
        
        for _ in range(n_samples):
            visible, _ = self._gibbs_sampling(initial_visible[0], n_steps=n_gibbs_steps)
            samples.append(visible)
        
        return np.array(samples)


class TemperatureScheduler:
    """
    Temperature-based learning rate scheduling
    
    Inspired by simulated annealing temperature schedules
    """
    
    def __init__(self, schedule_type: str = 'exponential', **kwargs):
        """
        Initialize temperature scheduler
        
        Args:
            schedule_type: 'exponential', 'linear', 'cosine', 'adaptive'
            **kwargs: Schedule-specific parameters
        """
        self.schedule_type = schedule_type
        self.kwargs = kwargs
        
        if schedule_type == 'exponential':
            self.initial_temp = kwargs.get('initial_temp', 1.0)
            self.decay_rate = kwargs.get('decay_rate', 0.95)
        elif schedule_type == 'linear':
            self.initial_temp = kwargs.get('initial_temp', 1.0)
            self.final_temp = kwargs.get('final_temp', 0.01)
            self.total_steps = kwargs.get('total_steps', 1000)
        elif schedule_type == 'cosine':
            self.initial_temp = kwargs.get('initial_temp', 1.0)
            self.final_temp = kwargs.get('final_temp', 0.01)
            self.total_steps = kwargs.get('total_steps', 1000)
        elif schedule_type == 'adaptive':
            self.initial_temp = kwargs.get('initial_temp', 1.0)
            self.patience = kwargs.get('patience', 10)
            self.factor = kwargs.get('factor', 0.5)
            self.min_temp = kwargs.get('min_temp', 0.01)
            self.best_metric = float('inf')
            self.no_improve_count = 0
    
    def get_temperature(self, step: int) -> float:
        """Get temperature at given step"""
        if self.schedule_type == 'exponential':
            return self.initial_temp * (self.decay_rate ** step)
        elif self.schedule_type == 'linear':
            return self.initial_temp - (self.initial_temp - self.final_temp) * \
                   (step / self.total_steps)
        elif self.schedule_type == 'cosine':
            return self.final_temp + (self.initial_temp - self.final_temp) * \
                   0.5 * (1 + np.cos(np.pi * step / self.total_steps))
        else:
            return self.initial_temp
    
    def step(self, metric: float) -> float:
        """Adaptive temperature update"""
        if self.schedule_type == 'adaptive':
            if metric < self.best_metric:
                self.best_metric = metric
                self.no_improve_count = 0
            else:
                self.no_improve_count += 1
                if self.no_improve_count >= self.patience:
                    self.initial_temp = max(self.initial_temp * self.factor, self.min_temp)
                    self.no_improve_count = 0
            return self.initial_temp
        return self.get_temperature(0)


def entropy_regularization(probabilities: np.ndarray, temperature: float = 1.0) -> float:
    """
    Entropy-based regularization term
    
    Encourages diverse, non-peaked probability distributions
    """
    # Avoid log(0)
    probabilities = np.clip(probabilities, 1e-10, 1.0)
    entropy = -np.sum(probabilities * np.log(probabilities))
    return entropy / temperature


def free_energy(energy: float, entropy: float, temperature: float = 1.0) -> float:
    """
    Free energy: F = E - T*S
    
    Combines energy (objective) and entropy (diversity)
    """
    return energy - temperature * entropy
