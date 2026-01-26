"""
Self-Organization & Dissipative Structures - Inspired by Ilya Prigogine

Implements:
- Self-Organizing Maps (SOM)
- Emergent Behaviors
- Dissipative Structures
- Far-From-Equilibrium Systems
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple
import logging
from scipy.spatial.distance import cdist

logger = logging.getLogger(__name__)


class SelfOrganizingMap:
    """
    Self-Organizing Map (Kohonen Map)
    
    Unsupervised learning with self-organization
    """
    
    def __init__(
        self,
        map_shape: Tuple[int, int],
        input_dim: int,
        learning_rate: float = 0.5,
        sigma: float = 1.0,
        learning_rate_decay: float = 0.99,
        sigma_decay: float = 0.99
    ):
        """
        Initialize Self-Organizing Map
        
        Args:
            map_shape: (height, width) of the map
            input_dim: Dimension of input vectors
            learning_rate: Initial learning rate
            sigma: Initial neighborhood radius
            learning_rate_decay: Learning rate decay per iteration
            sigma_decay: Neighborhood radius decay per iteration
        """
        self.map_shape = map_shape
        self.input_dim = input_dim
        self.learning_rate = learning_rate
        self.initial_learning_rate = learning_rate
        self.sigma = sigma
        self.initial_sigma = sigma
        self.learning_rate_decay = learning_rate_decay
        self.sigma_decay = sigma_decay
        
        # Initialize weights randomly
        self.weights = np.random.random((map_shape[0], map_shape[1], input_dim))
        
        # Create coordinate grid for distance calculation
        self.coordinates = np.array([[i, j] for i in range(map_shape[0]) 
                                    for j in range(map_shape[1])])
        self.coordinates = self.coordinates.reshape(map_shape + (2,))
    
    def _find_best_matching_unit(self, input_vector: np.ndarray) -> Tuple[int, int]:
        """Find best matching unit (BMU)"""
        distances = np.sum((self.weights - input_vector) ** 2, axis=2)
        bmu_idx = np.unravel_index(np.argmin(distances), distances.shape)
        return bmu_idx
    
    def _neighborhood_function(self, distance: float) -> float:
        """Gaussian neighborhood function"""
        return np.exp(-(distance ** 2) / (2 * self.sigma ** 2))
    
    def fit(self, X: np.ndarray, epochs: int = 100, verbose: bool = False):
        """
        Train Self-Organizing Map
        
        Args:
            X: Training data
            epochs: Number of training epochs
            verbose: Print progress
        """
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            
            for idx in indices:
                input_vector = X[idx]
                
                # Find BMU
                bmu_i, bmu_j = self._find_best_matching_unit(input_vector)
                bmu_pos = np.array([bmu_i, bmu_j])
                
                # Update weights
                for i in range(self.map_shape[0]):
                    for j in range(self.map_shape[1]):
                        node_pos = np.array([i, j])
                        distance = np.linalg.norm(node_pos - bmu_pos)
                        
                        # Neighborhood influence
                        influence = self._neighborhood_function(distance)
                        
                        # Update weight
                        self.weights[i, j] += self.learning_rate * influence * \
                                           (input_vector - self.weights[i, j])
            
            # Decay learning rate and neighborhood
            self.learning_rate *= self.learning_rate_decay
            self.sigma *= self.sigma_decay
            
            if verbose and (epoch + 1) % 10 == 0:
                logger.info(f"Epoch {epoch + 1}/{epochs}, LR: {self.learning_rate:.4f}, "
                          f"Sigma: {self.sigma:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Find BMU for each input
        
        Args:
            X: Input data
        
        Returns:
            BMU coordinates for each input
        """
        bmus = []
        for x in X:
            bmu_i, bmu_j = self._find_best_matching_unit(x)
            bmus.append([bmu_i, bmu_j])
        return np.array(bmus)
    
    def get_umatrix(self) -> np.ndarray:
        """
        Get U-matrix (unified distance matrix)
        
        Shows distances between neighboring units
        """
        umatrix = np.zeros(self.map_shape)
        
        for i in range(self.map_shape[0]):
            for j in range(self.map_shape[1]):
                neighbors = []
                
                # Get neighbors
                if i > 0:
                    neighbors.append(self.weights[i - 1, j])
                if i < self.map_shape[0] - 1:
                    neighbors.append(self.weights[i + 1, j])
                if j > 0:
                    neighbors.append(self.weights[i, j - 1])
                if j < self.map_shape[1] - 1:
                    neighbors.append(self.weights[i, j + 1])
                
                if neighbors:
                    distances = [np.linalg.norm(self.weights[i, j] - n) for n in neighbors]
                    umatrix[i, j] = np.mean(distances)
        
        return umatrix


class EmergentBehaviorSystem:
    """
    System that exhibits emergent behaviors from local interactions
    """
    
    def __init__(self, n_agents: int, interaction_radius: float = 1.0):
        """
        Initialize emergent behavior system
        
        Args:
            n_agents: Number of agents
            interaction_radius: Radius of interaction
        """
        self.n_agents = n_agents
        self.interaction_radius = interaction_radius
        
        # Initialize agent positions and states
        self.positions = np.random.random((n_agents, 2))
        self.velocities = np.random.normal(0, 0.1, (n_agents, 2))
        self.states = np.random.random(n_agents)
    
    def update(self, dt: float = 0.1):
        """
        Update system state (local interactions)
        
        Args:
            dt: Time step
        """
        # Calculate distances
        distances = cdist(self.positions, self.positions)
        
        # Local interactions (simplified: alignment rule)
        new_velocities = self.velocities.copy()
        
        for i in range(self.n_agents):
            # Find neighbors within interaction radius
            neighbors = np.where((distances[i] < self.interaction_radius) & 
                               (distances[i] > 0))[0]
            
            if len(neighbors) > 0:
                # Align with neighbors (emergent behavior)
                neighbor_velocities = self.velocities[neighbors]
                average_velocity = np.mean(neighbor_velocities, axis=0)
                
                # Update velocity (with inertia)
                new_velocities[i] = 0.7 * self.velocities[i] + 0.3 * average_velocity
        
        self.velocities = new_velocities
        
        # Update positions
        self.positions += self.velocities * dt
        
        # Boundary conditions (wrap around)
        self.positions = self.positions % 1.0
    
    def get_emergent_properties(self) -> Dict[str, float]:
        """
        Measure emergent properties
        
        Returns:
            Dictionary of emergent metrics
        """
        # Order parameter (alignment)
        velocity_norms = np.linalg.norm(self.velocities, axis=1)
        if np.sum(velocity_norms) > 0:
            normalized_velocities = self.velocities / (velocity_norms[:, np.newaxis] + 1e-10)
            order_parameter = np.linalg.norm(np.mean(normalized_velocities, axis=0))
        else:
            order_parameter = 0.0
        
        # Clustering (average distance to nearest neighbor)
        distances = cdist(self.positions, self.positions)
        np.fill_diagonal(distances, np.inf)
        nearest_distances = np.min(distances, axis=1)
        clustering = np.mean(nearest_distances)
        
        return {
            'order_parameter': order_parameter,
            'clustering': clustering,
            'average_speed': np.mean(velocity_norms)
        }


class DissipativeStructure:
    """
    Dissipative structure - maintained by energy flow
    """
    
    def __init__(self, structure_size: int, energy_input: float = 1.0):
        """
        Initialize dissipative structure
        
        Args:
            structure_size: Size of structure
            energy_input: Energy input rate
        """
        self.structure_size = structure_size
        self.energy_input = energy_input
        
        # State variables
        self.state = np.random.random(structure_size)
        self.energy = np.ones(structure_size) * energy_input
        self.entropy = np.zeros(structure_size)
    
    def update(self, dt: float = 0.01):
        """
        Update dissipative structure
        
        Args:
            dt: Time step
        """
        # Energy flow
        energy_flow = self.energy_input - 0.1 * self.state  # Dissipation
        
        # State evolution (far from equilibrium)
        state_change = energy_flow * (1 - self.state) - 0.05 * self.state
        
        self.state += state_change * dt
        self.state = np.clip(self.state, 0, 1)
        
        # Energy update
        self.energy = self.energy_input - 0.1 * self.state
        
        # Entropy (measure of disorder)
        self.entropy = -self.state * np.log(self.state + 1e-10) - \
                      (1 - self.state) * np.log(1 - self.state + 1e-10)
    
    def is_stable(self) -> bool:
        """Check if structure is stable (far from equilibrium)"""
        # Structure is stable if entropy production balances dissipation
        entropy_production = np.sum(self.entropy)
        energy_dissipation = np.sum(0.1 * self.state)
        
        return abs(entropy_production - energy_dissipation) < 0.1
