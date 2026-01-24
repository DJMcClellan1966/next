"""
Probabilistic Reasoning

Implements:
- Bayesian Networks
- Markov Chains
- Hidden Markov Models (HMM)
- Probabilistic Inference
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Set
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class BayesianNetwork:
    """
    Bayesian Network
    
    Directed acyclic graph representing probabilistic relationships
    """
    
    def __init__(self):
        """Initialize Bayesian network"""
        self.nodes = {}  # node -> {parents: [], cpt: {}}
        self.edges = []  # List of (parent, child) tuples
    
    def add_node(self, node: str, parents: List[str] = None,
                cpt: Dict[Tuple, float] = None):
        """
        Add node to network
        
        Parameters
        ----------
        node : str
            Node name
        parents : list of str
            Parent nodes
        cpt : dict
            Conditional probability table {(parent_values,): probability}
        """
        if parents is None:
            parents = []
        if cpt is None:
            cpt = {}
        
        self.nodes[node] = {
            'parents': parents,
            'cpt': cpt
        }
        
        # Add edges
        for parent in parents:
            self.edges.append((parent, node))
    
    def get_probability(self, node: str, value: bool, evidence: Dict[str, bool] = None) -> float:
        """
        Get probability of node given evidence
        
        Parameters
        ----------
        node : str
            Node name
        value : bool
            Node value
        evidence : dict
            Evidence (other node values)
            
        Returns
        -------
        probability : float
            Probability
        """
        if evidence is None:
            evidence = {}
        
        node_info = self.nodes[node]
        parents = node_info['parents']
        cpt = node_info['cpt']
        
        if not parents:
            # Root node - use prior
            if value:
                return cpt.get((), 0.5)  # Default 0.5 if not specified
            else:
                return 1 - cpt.get((), 0.5)
        else:
            # Get parent values from evidence
            parent_values = tuple(evidence.get(p, False) for p in parents)
            
            if value:
                return cpt.get(parent_values, 0.5)
            else:
                return 1 - cpt.get(parent_values, 0.5)
    
    def infer(self, query: str, evidence: Dict[str, bool]) -> float:
        """
        Perform inference (simplified)
        
        Parameters
        ----------
        query : str
            Query node
        evidence : dict
            Evidence
            
        Returns
        -------
        probability : float
            Probability of query given evidence
        """
        # Simple inference using chain rule
        # P(query|evidence) = P(query, evidence) / P(evidence)
        
        # For simplicity, use direct CPT lookup
        return self.get_probability(query, True, evidence)


class MarkovChain:
    """
    Markov Chain
    
    Sequence of states with Markov property
    """
    
    def __init__(self, transition_matrix: np.ndarray, initial_distribution: np.ndarray = None):
        """
        Initialize Markov chain
        
        Parameters
        ----------
        transition_matrix : array, shape (n_states, n_states)
            Transition probability matrix
        initial_distribution : array, optional
            Initial state distribution
        """
        self.transition_matrix = np.asarray(transition_matrix)
        self.n_states = self.transition_matrix.shape[0]
        
        if initial_distribution is None:
            self.initial_distribution = np.ones(self.n_states) / self.n_states
        else:
            self.initial_distribution = np.asarray(initial_distribution)
    
    def get_next_state(self, current_state: int) -> int:
        """
        Sample next state
        
        Parameters
        ----------
        current_state : int
            Current state
            
        Returns
        -------
        next_state : int
            Next state
        """
        probs = self.transition_matrix[current_state]
        return np.random.choice(self.n_states, p=probs)
    
    def generate_sequence(self, length: int) -> List[int]:
        """
        Generate state sequence
        
        Parameters
        ----------
        length : int
            Sequence length
            
        Returns
        -------
        sequence : list
            State sequence
        """
        sequence = []
        
        # Initial state
        state = np.random.choice(self.n_states, p=self.initial_distribution)
        sequence.append(state)
        
        # Generate sequence
        for _ in range(length - 1):
            state = self.get_next_state(state)
            sequence.append(state)
        
        return sequence
    
    def get_stationary_distribution(self) -> np.ndarray:
        """
        Get stationary distribution
        
        Returns
        -------
        distribution : array
            Stationary distribution
        """
        # Find eigenvector corresponding to eigenvalue 1
        eigenvalues, eigenvectors = np.linalg.eig(self.transition_matrix.T)
        idx = np.argmax(np.real(eigenvalues))
        stationary = np.real(eigenvectors[:, idx])
        stationary = stationary / np.sum(stationary)
        return stationary


class HMM:
    """
    Hidden Markov Model
    
    Markov chain with hidden states and observable emissions
    """
    
    def __init__(self, n_states: int, n_observations: int,
                transition_matrix: np.ndarray = None,
                emission_matrix: np.ndarray = None,
                initial_distribution: np.ndarray = None):
        """
        Initialize HMM
        
        Parameters
        ----------
        n_states : int
            Number of hidden states
        n_observations : int
            Number of possible observations
        transition_matrix : array, optional
            State transition matrix
        emission_matrix : array, optional
            Emission probability matrix
        initial_distribution : array, optional
            Initial state distribution
        """
        self.n_states = n_states
        self.n_observations = n_observations
        
        if transition_matrix is None:
            self.transition_matrix = np.ones((n_states, n_states)) / n_states
        else:
            self.transition_matrix = np.asarray(transition_matrix)
        
        if emission_matrix is None:
            self.emission_matrix = np.ones((n_states, n_observations)) / n_observations
        else:
            self.emission_matrix = np.asarray(emission_matrix)
        
        if initial_distribution is None:
            self.initial_distribution = np.ones(n_states) / n_states
        else:
            self.initial_distribution = np.asarray(initial_distribution)
    
    def forward(self, observations: List[int]) -> np.ndarray:
        """
        Forward algorithm (computes P(observations))
        
        Parameters
        ----------
        observations : list
            Observation sequence
            
        Returns
        -------
        alpha : array
            Forward probabilities
        """
        T = len(observations)
        alpha = np.zeros((T, self.n_states))
        
        # Initialization
        alpha[0] = self.initial_distribution * self.emission_matrix[:, observations[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                alpha[t, j] = self.emission_matrix[j, observations[t]] * \
                            np.sum(alpha[t-1] * self.transition_matrix[:, j])
        
        return alpha
    
    def viterbi(self, observations: List[int]) -> List[int]:
        """
        Viterbi algorithm (most likely state sequence)
        
        Parameters
        ----------
        observations : list
            Observation sequence
            
        Returns
        -------
        states : list
            Most likely state sequence
        """
        T = len(observations)
        viterbi = np.zeros((T, self.n_states))
        backpointer = np.zeros((T, self.n_states), dtype=int)
        
        # Initialization
        viterbi[0] = self.initial_distribution * self.emission_matrix[:, observations[0]]
        
        # Recursion
        for t in range(1, T):
            for j in range(self.n_states):
                probs = viterbi[t-1] * self.transition_matrix[:, j] * \
                       self.emission_matrix[j, observations[t]]
                viterbi[t, j] = np.max(probs)
                backpointer[t, j] = np.argmax(probs)
        
        # Backtracking
        states = [np.argmax(viterbi[T-1])]
        for t in range(T-1, 0, -1):
            states.insert(0, backpointer[t, states[0]])
        
        return states


class Inference:
    """
    Probabilistic Inference Methods
    
    Various inference algorithms
    """
    
    @staticmethod
    def variable_elimination(bn: BayesianNetwork, query: str,
                           evidence: Dict[str, bool]) -> float:
        """
        Variable elimination for inference
        
        Parameters
        ----------
        bn : BayesianNetwork
            Bayesian network
        query : str
            Query variable
        evidence : dict
            Evidence
            
        Returns
        -------
        probability : float
            Probability
        """
        # Simplified variable elimination
        # In practice, more sophisticated algorithm needed
        return bn.infer(query, evidence)
    
    @staticmethod
    def gibbs_sampling(bn: BayesianNetwork, query: str, evidence: Dict[str, bool],
                      n_samples: int = 1000) -> float:
        """
        Gibbs sampling for inference
        
        Parameters
        ----------
        bn : BayesianNetwork
            Bayesian network
        query : str
            Query variable
        evidence : dict
            Evidence
        n_samples : int
            Number of samples
            
        Returns
        -------
        probability : float
            Estimated probability
        """
        # Simplified Gibbs sampling
        # Initialize non-evidence variables randomly
        non_evidence = [n for n in bn.nodes.keys() if n not in evidence]
        
        # Sample
        query_true = 0
        for _ in range(n_samples):
            # Sample each non-evidence variable
            for node in non_evidence:
                if node == query:
                    # Sample query variable
                    prob = bn.get_probability(node, True, evidence)
                    if np.random.random() < prob:
                        evidence[query] = True
                        query_true += 1
                    else:
                        evidence[query] = False
        
        return query_true / n_samples if n_samples > 0 else 0.0
