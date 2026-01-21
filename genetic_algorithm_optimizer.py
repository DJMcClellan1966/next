"""
Russell/Norvig Genetic Algorithms for Model Selection
Evolutionary algorithm for hyperparameter optimization

Features:
- Genetic algorithm for hyperparameter search
- Population-based optimization
- Crossover and mutation operators
- Selection strategies (tournament, roulette wheel)
- Elitism
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import numpy as np
from collections import defaultdict
import warnings

sys.path.insert(0, str(Path(__file__).parent))

# Try to import sklearn
try:
    from sklearn.model_selection import cross_val_score, KFold, StratifiedKFold
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Install with: pip install scikit-learn")


class GeneticAlgorithmOptimizer:
    """
    Genetic Algorithm for hyperparameter optimization
    
    Uses evolutionary principles: selection, crossover, mutation
    """
    
    def __init__(
        self,
        population_size: int = 50,
        n_generations: int = 20,
        crossover_rate: float = 0.8,
        mutation_rate: float = 0.1,
        elitism_rate: float = 0.1,
        selection_method: str = 'tournament',
        tournament_size: int = 3,
        random_state: int = 42
    ):
        """
        Args:
            population_size: Size of population
            n_generations: Number of generations
            crossover_rate: Probability of crossover
            mutation_rate: Probability of mutation
            elitism_rate: Fraction of best individuals to keep
            selection_method: 'tournament' or 'roulette'
            tournament_size: Size of tournament for selection
            random_state: Random seed
        """
        self.population_size = population_size
        self.n_generations = n_generations
        self.crossover_rate = crossover_rate
        self.mutation_rate = mutation_rate
        self.elitism_rate = elitism_rate
        self.selection_method = selection_method
        self.tournament_size = tournament_size
        self.random_state = random_state
        np.random.seed(random_state)
        
        self.best_params_ = None
        self.best_score_ = -np.inf
        self.optimization_history_ = []
    
    def _initialize_population(self, param_space: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Initialize random population"""
        population = []
        for _ in range(self.population_size):
            individual = {}
            for name, values in param_space.items():
                if isinstance(values, tuple) and len(values) == 2:
                    min_val, max_val = values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        individual[name] = np.random.randint(min_val, max_val + 1)
                    else:
                        individual[name] = np.random.uniform(min_val, max_val)
                elif isinstance(values, list):
                    individual[name] = np.random.choice(values)
                else:
                    individual[name] = values
            population.append(individual)
        return population
    
    def _evaluate_individual(
        self,
        model_class: type,
        individual: Dict[str, Any],
        X: np.ndarray,
        y: np.ndarray,
        scoring: Optional[str] = None,
        cv: int = 5
    ) -> float:
        """Evaluate individual using cross-validation"""
        if not SKLEARN_AVAILABLE:
            return 0.0
        
        try:
            model = model_class(**individual)
        except:
            return -np.inf
        
        if scoring is None:
            scoring = 'accuracy' if len(np.unique(y)) < 10 else 'r2'
        
        if len(np.unique(y)) < 10:
            cv_splitter = StratifiedKFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        else:
            cv_splitter = KFold(n_splits=cv, shuffle=True, random_state=self.random_state)
        
        try:
            scores = cross_val_score(model, X, y, cv=cv_splitter, scoring=scoring, n_jobs=-1)
            return np.mean(scores)
        except:
            return -np.inf
    
    def _evaluate_population(
        self,
        model_class: type,
        population: List[Dict[str, Any]],
        X: np.ndarray,
        y: np.ndarray,
        scoring: Optional[str],
        cv: int
    ) -> List[float]:
        """Evaluate entire population"""
        return [self._evaluate_individual(model_class, ind, X, y, scoring, cv) for ind in population]
    
    def _select_parents(
        self,
        population: List[Dict[str, Any]],
        fitness: List[float]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Select two parents for crossover"""
        if self.selection_method == 'tournament':
            # Tournament selection
            def tournament_select():
                tournament_indices = np.random.choice(len(population), self.tournament_size, replace=False)
                tournament_fitness = [fitness[i] for i in tournament_indices]
                winner_idx = tournament_indices[np.argmax(tournament_fitness)]
                return population[winner_idx]
            
            return tournament_select(), tournament_select()
        
        elif self.selection_method == 'roulette':
            # Roulette wheel selection
            fitness_array = np.array(fitness)
            fitness_array = fitness_array - fitness_array.min() + 1e-10  # Make positive
            probs = fitness_array / fitness_array.sum()
            
            indices = np.random.choice(len(population), size=2, p=probs, replace=True)
            return population[indices[0]], population[indices[1]]
        
        else:
            # Random selection
            indices = np.random.choice(len(population), size=2, replace=False)
            return population[indices[0]], population[indices[1]]
    
    def _crossover(
        self,
        parent1: Dict[str, Any],
        parent2: Dict[str, Any],
        param_space: Dict[str, Any]
    ) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        """Crossover two parents to create offspring"""
        if np.random.random() > self.crossover_rate:
            return parent1.copy(), parent2.copy()
        
        child1 = {}
        child2 = {}
        
        for name in param_space.keys():
            if np.random.random() < 0.5:
                child1[name] = parent1[name]
                child2[name] = parent2[name]
            else:
                child1[name] = parent2[name]
                child2[name] = parent1[name]
        
        return child1, child2
    
    def _mutate(
        self,
        individual: Dict[str, Any],
        param_space: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Mutate individual"""
        mutated = individual.copy()
        
        for name, values in param_space.items():
            if np.random.random() < self.mutation_rate:
                # Mutate this parameter
                if isinstance(values, tuple) and len(values) == 2:
                    min_val, max_val = values
                    if isinstance(min_val, int) and isinstance(max_val, int):
                        mutated[name] = np.random.randint(min_val, max_val + 1)
                    else:
                        mutated[name] = np.random.uniform(min_val, max_val)
                elif isinstance(values, list):
                    mutated[name] = np.random.choice(values)
        
        return mutated
    
    def optimize(
        self,
        model_class: type,
        X: np.ndarray,
        y: np.ndarray,
        param_space: Dict[str, Any],
        scoring: Optional[str] = None,
        cv: int = 5
    ) -> Dict[str, Any]:
        """
        Optimize hyperparameters using genetic algorithm
        
        Args:
            model_class: Model class
            X: Features
            y: Labels
            param_space: Parameter space
            scoring: Scoring metric
            cv: Cross-validation folds
            
        Returns:
            Dictionary with best parameters, best score, optimization history
        """
        X = np.asarray(X)
        y = np.asarray(y)
        
        # Initialize population
        population = self._initialize_population(param_space)
        
        # Evaluate initial population
        fitness = self._evaluate_population(model_class, population, X, y, scoring, cv)
        
        # Track best
        best_idx = np.argmax(fitness)
        self.best_params_ = population[best_idx].copy()
        self.best_score_ = fitness[best_idx]
        
        # Evolution loop
        for generation in range(self.n_generations):
            # Sort by fitness
            sorted_indices = np.argsort(fitness)[::-1]
            population = [population[i] for i in sorted_indices]
            fitness = [fitness[i] for i in sorted_indices]
            
            # Elitism: keep best individuals
            n_elite = int(self.population_size * self.elitism_rate)
            elite = population[:n_elite]
            elite_fitness = fitness[:n_elite]
            
            # Create new population
            new_population = elite.copy()
            new_fitness = elite_fitness.copy()
            
            # Generate offspring
            while len(new_population) < self.population_size:
                # Select parents
                parent1, parent2 = self._select_parents(population, fitness)
                
                # Crossover
                child1, child2 = self._crossover(parent1, parent2, param_space)
                
                # Mutate
                child1 = self._mutate(child1, param_space)
                child2 = self._mutate(child2, param_space)
                
                # Add to new population
                new_population.append(child1)
                if len(new_population) < self.population_size:
                    new_population.append(child2)
            
            # Evaluate new population
            new_fitness = self._evaluate_population(model_class, new_population, X, y, scoring, cv)
            
            # Update population
            population = new_population
            fitness = new_fitness
            
            # Update best
            best_idx = np.argmax(fitness)
            if fitness[best_idx] > self.best_score_:
                self.best_params_ = population[best_idx].copy()
                self.best_score_ = fitness[best_idx]
            
            # Record history
            self.optimization_history_.append({
                'generation': generation,
                'best_score': self.best_score_,
                'mean_score': np.mean(fitness),
                'std_score': np.std(fitness)
            })
        
        return {
            'best_params': self.best_params_,
            'best_score': float(self.best_score_),
            'n_generations': self.n_generations,
            'optimization_history': self.optimization_history_,
            'method': 'genetic_algorithm'
        }
