"""
Evolutionary Algorithms - Inspired by Charles Darwin

Implements:
- Genetic Algorithms (GA)
- Evolutionary Strategies (ES)
- Differential Evolution (DE)
- Neuroevolution (NE)
- Evolutionary Feature Selection
"""
import numpy as np
from typing import List, Dict, Any, Callable, Optional, Tuple
import random
import logging
from copy import deepcopy

logger = logging.getLogger(__name__)


class GeneticAlgorithm:
    """
    Genetic Algorithm for optimization
    
    Inspired by natural selection: selection, crossover, mutation
    """
    
    def __init__(
        self,
        fitness_function: Callable,
        gene_ranges: List[Tuple[float, float]],
        population_size: int = 50,
        mutation_rate: float = 0.1,
        crossover_rate: float = 0.8,
        elitism: bool = True,
        elite_size: int = 2,
        selection_method: str = 'tournament',
        tournament_size: int = 3,
        max_generations: int = 100,
        convergence_threshold: float = 1e-6
    ):
        """
        Initialize Genetic Algorithm
        
        Args:
            fitness_function: Function to maximize (takes individual as input)
            gene_ranges: List of (min, max) tuples for each gene
            population_size: Size of population
            mutation_rate: Probability of mutation
            crossover_rate: Probability of crossover
            elitism: Whether to preserve best individuals
            elite_size: Number of elite individuals to preserve
            selection_method: 'tournament' or 'roulette'
            tournament_size: Size of tournament for selection
            max_generations: Maximum number of generations
            convergence_threshold: Convergence threshold
        """
        self.fitness_function = fitness_function
        self.gene_ranges = gene_ranges
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.crossover_rate = crossover_rate
        self.elitism = elitism
        self.elite_size = elite_size
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.max_generations = max_generations
        self.convergence_threshold = convergence_threshold
        
        self.num_genes = len(gene_ranges)
        self.population = None
        self.fitness_history = []
        self.best_individual = None
        self.best_fitness = float('-inf')
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = []
            for min_val, max_val in self.gene_ranges:
                individual.append(random.uniform(min_val, max_val))
            population.append(individual)
        return np.array(population)
    
    def _evaluate_population(self, population: np.ndarray) -> np.ndarray:
        """Evaluate fitness of all individuals"""
        fitness_scores = []
        for individual in population:
            try:
                fitness = self.fitness_function(individual)
                fitness_scores.append(fitness)
            except Exception as e:
                logger.warning(f"Fitness evaluation failed: {e}")
                fitness_scores.append(float('-inf'))
        return np.array(fitness_scores)
    
    def _tournament_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Tournament selection"""
        selected = []
        for _ in range(self.population_size):
            tournament_indices = random.sample(range(len(population)), self.tournament_size)
            tournament_fitness = fitness[tournament_indices]
            winner_idx = tournament_indices[np.argmax(tournament_fitness)]
            selected.append(population[winner_idx])
        return np.array(selected)
    
    def _roulette_selection(self, population: np.ndarray, fitness: np.ndarray) -> np.ndarray:
        """Roulette wheel selection"""
        # Normalize fitness to positive values
        min_fitness = np.min(fitness)
        if min_fitness < 0:
            fitness = fitness - min_fitness + 1e-10
        
        # Calculate probabilities
        total_fitness = np.sum(fitness)
        if total_fitness == 0:
            return self._tournament_selection(population, fitness)
        
        probabilities = fitness / total_fitness
        
        # Select based on probabilities
        selected_indices = np.random.choice(
            len(population),
            size=self.population_size,
            p=probabilities
        )
        return population[selected_indices]
    
    def _crossover(self, parent1: np.ndarray, parent2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        """Single-point crossover"""
        if random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        crossover_point = random.randint(1, self.num_genes - 1)
        child1 = np.concatenate([parent1[:crossover_point], parent2[crossover_point:]])
        child2 = np.concatenate([parent2[:crossover_point], parent1[crossover_point:]])
        return child1, child2
    
    def _mutate(self, individual: np.ndarray) -> np.ndarray:
        """Gaussian mutation"""
        mutated = individual.copy()
        for i, (min_val, max_val) in enumerate(self.gene_ranges):
            if random.random() < self.mutation_rate:
                # Gaussian mutation with range constraint
                std = (max_val - min_val) * 0.1
                mutation = np.random.normal(0, std)
                mutated[i] = np.clip(mutated[i] + mutation, min_val, max_val)
        return mutated
    
    def evolve(self) -> Dict[str, Any]:
        """
        Run genetic algorithm evolution
        
        Returns:
            Dictionary with best individual, fitness, and history
        """
        # Initialize population
        self.population = self._initialize_population()
        
        # Evaluate initial population
        fitness = self._evaluate_population(self.population)
        
        # Track best
        best_idx = np.argmax(fitness)
        self.best_individual = self.population[best_idx].copy()
        self.best_fitness = fitness[best_idx]
        self.fitness_history = [self.best_fitness]
        
        # Evolution loop
        for generation in range(self.max_generations):
            # Selection
            if self.selection_method == 'tournament':
                selected = self._tournament_selection(self.population, fitness)
            else:
                selected = self._roulette_selection(self.population, fitness)
            
            # Crossover and mutation
            new_population = []
            for i in range(0, len(selected) - 1, 2):
                parent1, parent2 = selected[i], selected[i + 1]
                child1, child2 = self._crossover(parent1, parent2)
                child1 = self._mutate(child1)
                child2 = self._mutate(child2)
                new_population.extend([child1, child2])
            
            # Add last individual if odd population size
            if len(new_population) < self.population_size:
                last = self._mutate(selected[-1])
                new_population.append(last)
            
            new_population = np.array(new_population[:self.population_size])
            
            # Elitism: preserve best individuals
            if self.elitism:
                elite_indices = np.argsort(fitness)[-self.elite_size:]
                elite = self.population[elite_indices]
                # Replace worst individuals with elite
                new_fitness = self._evaluate_population(new_population)
                worst_indices = np.argsort(new_fitness)[:self.elite_size]
                new_population[worst_indices] = elite
            
            # Update population
            self.population = new_population
            fitness = self._evaluate_population(self.population)
            
            # Track best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self.best_fitness:
                self.best_individual = self.population[best_idx].copy()
                self.best_fitness = fitness[best_idx]
            
            self.fitness_history.append(self.best_fitness)
            
            # Check convergence
            if len(self.fitness_history) > 10:
                recent_improvement = abs(self.fitness_history[-1] - self.fitness_history[-10])
                if recent_improvement < self.convergence_threshold:
                    logger.info(f"Converged at generation {generation}")
                    break
        
        return {
            'best_individual': self.best_individual,
            'best_fitness': self.best_fitness,
            'fitness_history': self.fitness_history,
            'generations': len(self.fitness_history)
        }


class DifferentialEvolution:
    """
    Differential Evolution - Global optimization algorithm
    """
    
    def __init__(
        self,
        fitness_function: Callable,
        bounds: List[Tuple[float, float]],
        population_size: int = 50,
        F: float = 0.5,  # Differential weight
        CR: float = 0.7,  # Crossover probability
        max_generations: int = 100
    ):
        """
        Initialize Differential Evolution
        
        Args:
            fitness_function: Function to minimize
            bounds: List of (min, max) tuples for each dimension
            population_size: Size of population
            F: Differential weight (typically 0.5-1.0)
            CR: Crossover probability (typically 0.7-0.9)
            max_generations: Maximum generations
        """
        self.fitness_function = fitness_function
        self.bounds = bounds
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.max_generations = max_generations
        self.dim = len(bounds)
    
    def _initialize_population(self) -> np.ndarray:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = [random.uniform(min_val, max_val) for min_val, max_val in self.bounds]
            population.append(individual)
        return np.array(population)
    
    def _mutate(self, population: np.ndarray, target_idx: int) -> np.ndarray:
        """Mutation: v = x_r1 + F * (x_r2 - x_r3)"""
        # Select three distinct random indices
        candidates = [i for i in range(self.population_size) if i != target_idx]
        r1, r2, r3 = random.sample(candidates, 3)
        
        mutant = population[r1] + self.F * (population[r2] - population[r3])
        
        # Apply bounds
        for i, (min_val, max_val) in enumerate(self.bounds):
            mutant[i] = np.clip(mutant[i], min_val, max_val)
        
        return mutant
    
    def _crossover(self, target: np.ndarray, mutant: np.ndarray) -> np.ndarray:
        """Binomial crossover"""
        trial = target.copy()
        j_rand = random.randint(0, self.dim - 1)  # Ensure at least one dimension changes
        
        for j in range(self.dim):
            if random.random() < self.CR or j == j_rand:
                trial[j] = mutant[j]
        
        return trial
    
    def optimize(self) -> Dict[str, Any]:
        """
        Run differential evolution
        
        Returns:
            Dictionary with best solution and fitness
        """
        population = self._initialize_population()
        
        # Evaluate initial population
        fitness = np.array([self.fitness_function(ind) for ind in population])
        best_idx = np.argmin(fitness)
        best_individual = population[best_idx].copy()
        best_fitness = fitness[best_idx]
        fitness_history = [best_fitness]
        
        for generation in range(self.max_generations):
            new_population = []
            new_fitness = []
            
            for i in range(self.population_size):
                # Mutation
                mutant = self._mutate(population, i)
                
                # Crossover
                trial = self._crossover(population[i], mutant)
                
                # Selection
                trial_fitness = self.fitness_function(trial)
                if trial_fitness < fitness[i]:
                    new_population.append(trial)
                    new_fitness.append(trial_fitness)
                else:
                    new_population.append(population[i])
                    new_fitness.append(fitness[i])
            
            population = np.array(new_population)
            fitness = np.array(new_fitness)
            
            # Update best
            best_idx = np.argmin(fitness)
            if fitness[best_idx] < best_fitness:
                best_individual = population[best_idx].copy()
                best_fitness = fitness[best_idx]
            
            fitness_history.append(best_fitness)
        
        return {
            'best_individual': best_individual,
            'best_fitness': best_fitness,
            'fitness_history': fitness_history
        }


def evolutionary_feature_selection(
    X: np.ndarray,
    y: np.ndarray,
    model: Any,
    n_features: int,
    population_size: int = 50,
    max_generations: int = 50,
    cv_folds: int = 5
) -> Dict[str, Any]:
    """
    Evolutionary feature selection using genetic algorithm
    
    Args:
        X: Feature matrix
        y: Target vector
        model: Scikit-learn compatible model
        n_features: Number of features to select
        population_size: GA population size
        max_generations: Maximum generations
        cv_folds: Cross-validation folds
    
    Returns:
        Dictionary with selected features and scores
    """
    from sklearn.model_selection import cross_val_score
    
    n_total_features = X.shape[1]
    
    def fitness_function(individual: np.ndarray) -> float:
        """Fitness: CV score with selected features"""
        # Convert to binary feature mask
        feature_mask = (individual > 0.5).astype(bool)
        
        if np.sum(feature_mask) == 0:
            return -1.0  # Penalize no features
        
        if np.sum(feature_mask) > n_features:
            return -0.5  # Penalize too many features
        
        # Select features
        X_selected = X[:, feature_mask]
        
        # Cross-validation score
        try:
            scores = cross_val_score(model, X_selected, y, cv=cv_folds, scoring='accuracy')
            return np.mean(scores)
        except:
            return -1.0
    
    # Initialize GA
    gene_ranges = [(0.0, 1.0)] * n_total_features
    ga = GeneticAlgorithm(
        fitness_function=fitness_function,
        gene_ranges=gene_ranges,
        population_size=population_size,
        max_generations=max_generations,
        mutation_rate=0.1,
        crossover_rate=0.8
    )
    
    # Evolve
    result = ga.evolve()
    
    # Get selected features
    feature_mask = (result['best_individual'] > 0.5).astype(bool)
    selected_features = np.where(feature_mask)[0].tolist()
    
    return {
        'selected_features': selected_features,
        'feature_mask': feature_mask,
        'fitness': result['best_fitness'],
        'fitness_history': result['fitness_history']
    }
