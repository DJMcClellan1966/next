"""
Parallel Universes / Multiverse Processing

Inspired by: "The Many-Worlds Interpretation", "Rick and Morty", "Fringe"

Implements:
- Parallel Universe Creation
- Multiverse Ensembles
- Decision Branching
- Universe Selection
- Inter-Universe Communication
"""
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
import copy
from concurrent.futures import ThreadPoolExecutor, as_completed

logger = logging.getLogger(__name__)


class ParallelUniverse:
    """
    Parallel Universe - Alternative reality for experimentation
    """
    
    def __init__(
        self,
        universe_id: str,
        initial_state: Dict[str, Any],
        model: Any,
        random_seed: Optional[int] = None
    ):
        """
        Initialize parallel universe
        
        Args:
            universe_id: Unique universe identifier
            initial_state: Initial state of universe
            model: Model in this universe
            random_seed: Random seed for reproducibility
        """
        self.universe_id = universe_id
        self.state = initial_state.copy()
        self.model = copy.deepcopy(model) if hasattr(model, 'copy') else model
        self.random_seed = random_seed
        self.history = []
        self.metrics = {}
        
        if random_seed is not None:
            np.random.seed(random_seed)
    
    def evolve(self, n_steps: int = 10, action: Optional[Callable] = None):
        """
        Evolve universe forward
        
        Args:
            n_steps: Number of evolution steps
            action: Action to take at each step
        """
        for step in range(n_steps):
            if action:
                result = action(self.state, self.model)
                self.state.update(result)
            
            self.history.append({
                'step': step,
                'state': self.state.copy(),
                'timestamp': len(self.history)
            })
    
    def branch(self, branch_point: str, alternative_state: Dict[str, Any]) -> 'ParallelUniverse':
        """
        Create a branch universe from this one
        
        Args:
            branch_point: Point where universe branches
            alternative_state: Alternative state for branch
        
        Returns:
            New parallel universe
        """
        new_state = self.state.copy()
        new_state.update(alternative_state)
        new_state['branch_point'] = branch_point
        new_state['parent_universe'] = self.universe_id
        
        new_universe = ParallelUniverse(
            universe_id=f"{self.universe_id}_branch_{len(self.history)}",
            initial_state=new_state,
            model=self.model,
            random_seed=self.random_seed
        )
        
        return new_universe
    
    def evaluate(self, metric: Callable) -> float:
        """
        Evaluate universe using metric
        
        Args:
            metric: Evaluation function
        
        Returns:
            Metric value
        """
        score = metric(self.state, self.model)
        self.metrics['last_evaluation'] = score
        return score


class MultiverseProcessor:
    """
    Multiverse Processor - Manage parallel universes
    """
    
    def __init__(self, n_universes: int = 10):
        """
        Initialize multiverse processor
        
        Args:
            n_universes: Number of parallel universes
        """
        self.n_universes = n_universes
        self.universes = {}
        self.universe_network = {}  # Connections between universes
    
    def create_universes(
        self,
        initial_states: List[Dict[str, Any]],
        models: List[Any],
        random_seeds: Optional[List[int]] = None
    ) -> List[str]:
        """
        Create multiple parallel universes
        
        Args:
            initial_states: Initial states for each universe
            models: Models for each universe
            random_seeds: Random seeds for each universe
        
        Returns:
            List of universe IDs
        """
        universe_ids = []
        
        for i in range(min(self.n_universes, len(initial_states))):
            universe_id = f"universe_{i}"
            seed = random_seeds[i] if random_seeds and i < len(random_seeds) else None
            
            universe = ParallelUniverse(
                universe_id=universe_id,
                initial_state=initial_states[i],
                model=models[i] if i < len(models) else models[0],
                random_seed=seed
            )
            
            self.universes[universe_id] = universe
            self.universe_network[universe_id] = []
            universe_ids.append(universe_id)
        
        return universe_ids
    
    def parallel_experiment(
        self,
        experiment_func: Callable,
        n_universes: Optional[int] = None
    ) -> Dict[str, Any]:
        """
        Run experiment in parallel universes
        
        Args:
            experiment_func: Function to run in each universe
            n_universes: Number of universes to use (None = all)
        
        Returns:
            Results from all universes
        """
        universe_ids = list(self.universes.keys())
        if n_universes:
            universe_ids = universe_ids[:n_universes]
        
        results = {}
        
        # Run in parallel
        with ThreadPoolExecutor(max_workers=len(universe_ids)) as executor:
            futures = {
                executor.submit(experiment_func, self.universes[uid]): uid
                for uid in universe_ids
            }
            
            for future in as_completed(futures):
                universe_id = futures[future]
                try:
                    result = future.result()
                    results[universe_id] = result
                except Exception as e:
                    logger.error(f"Universe {universe_id} experiment failed: {e}")
                    results[universe_id] = {'error': str(e)}
        
        return results
    
    def multiverse_ensemble(
        self,
        X: np.ndarray,
        aggregation_method: str = 'mean'
    ) -> np.ndarray:
        """
        Ensemble predictions from multiple universes
        
        Args:
            X: Input data
            aggregation_method: 'mean', 'median', 'vote', 'weighted'
        
        Returns:
            Ensemble prediction
        """
        predictions = []
        weights = []
        
        for universe_id, universe in self.universes.items():
            try:
                if hasattr(universe.model, 'predict'):
                    pred = universe.model.predict(X)
                    predictions.append(pred)
                    
                    # Weight by universe quality
                    quality = universe.metrics.get('last_evaluation', 1.0)
                    weights.append(quality)
            except Exception as e:
                logger.warning(f"Universe {universe_id} prediction failed: {e}")
        
        if not predictions:
            raise ValueError("No valid predictions from universes")
        
        predictions = np.array(predictions)
        
        if aggregation_method == 'mean':
            if len(weights) > 0:
                weights = np.array(weights)
                weights = weights / weights.sum()
                return np.average(predictions, axis=0, weights=weights)
            return np.mean(predictions, axis=0)
        
        elif aggregation_method == 'median':
            return np.median(predictions, axis=0)
        
        elif aggregation_method == 'vote':
            # For classification
            from scipy.stats import mode
            return mode(predictions, axis=0)[0].flatten()
        
        else:  # weighted
            weights = np.array(weights)
            weights = weights / weights.sum()
            return np.average(predictions, axis=0, weights=weights)
    
    def select_best_universe(self, metric: Callable) -> str:
        """
        Select best universe based on metric
        
        Args:
            metric: Evaluation metric
        
        Returns:
            Best universe ID
        """
        best_universe = None
        best_score = float('-inf')
        
        for universe_id, universe in self.universes.items():
            score = universe.evaluate(metric)
            if score > best_score:
                best_score = score
                best_universe = universe_id
        
        return best_universe
    
    def collapse_universes(self, keep_universe: str):
        """
        Collapse all universes except one (quantum collapse)
        
        Args:
            keep_universe: Universe to keep
        """
        if keep_universe not in self.universes:
            raise ValueError(f"Universe {keep_universe} not found")
        
        # Keep only selected universe
        self.universes = {keep_universe: self.universes[keep_universe]}
        self.universe_network = {keep_universe: []}
    
    def branch_on_decision(
        self,
        universe_id: str,
        decision_point: str,
        alternatives: List[Dict[str, Any]]
    ) -> List[str]:
        """
        Branch universe on decision point
        
        Args:
            universe_id: Universe to branch
            decision_point: Decision point identifier
            alternatives: Alternative decisions
        
        Returns:
            List of new universe IDs
        """
        if universe_id not in self.universes:
            raise ValueError(f"Universe {universe_id} not found")
        
        parent_universe = self.universes[universe_id]
        new_universe_ids = []
        
        for i, alternative in enumerate(alternatives):
            branch_universe = parent_universe.branch(decision_point, alternative)
            new_id = f"{universe_id}_branch_{i}"
            branch_universe.universe_id = new_id
            
            self.universes[new_id] = branch_universe
            self.universe_network[new_id] = [universe_id]
            self.universe_network[universe_id].append(new_id)
            new_universe_ids.append(new_id)
        
        return new_universe_ids
    
    def communicate_between_universes(
        self,
        source_universe: str,
        target_universe: str,
        data: Any
    ):
        """
        Communicate between universes
        
        Args:
            source_universe: Source universe ID
            target_universe: Target universe ID
            data: Data to transfer
        """
        if source_universe not in self.universes or target_universe not in self.universes:
            raise ValueError("Universe not found")
        
        # Transfer knowledge/state
        source = self.universes[source_universe]
        target = self.universes[target_universe]
        
        # Share state information
        if 'shared_knowledge' not in target.state:
            target.state['shared_knowledge'] = {}
        
        target.state['shared_knowledge'][source_universe] = data
    
    def get_multiverse_status(self) -> Dict[str, Any]:
        """Get status of multiverse"""
        return {
            'n_universes': len(self.universes),
            'universe_ids': list(self.universes.keys()),
            'network_connections': {
                uid: len(connections)
                for uid, connections in self.universe_network.items()
            }
        }
