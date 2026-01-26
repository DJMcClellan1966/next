"""
Network Theory & Scale-Free Networks - Inspired by Albert-László Barabási

Implements:
- Scale-Free Network Generation
- Network Centrality Measures
- Community Detection
- Small-World Networks
- Network-Based Feature Importance
"""
import numpy as np
from typing import List, Dict, Any, Optional, Tuple, Set
import logging
from collections import defaultdict, deque

logger = logging.getLogger(__name__)


class ScaleFreeNetwork:
    """
    Scale-Free Network Generator
    
    Networks with power-law degree distribution
    """
    
    def __init__(self, n_nodes: int, m: int = 2):
        """
        Initialize scale-free network generator
        
        Args:
            n_nodes: Number of nodes
            m: Number of edges to attach from new node to existing nodes
        """
        self.n_nodes = n_nodes
        self.m = m
        self.adjacency_matrix = None
        self.edges = []
        self.degrees = None
    
    def generate(self) -> np.ndarray:
        """
        Generate scale-free network using Barabási-Albert model
        
        Returns:
            Adjacency matrix
        """
        n = self.n_nodes
        m = self.m
        
        # Initialize with m nodes fully connected
        adjacency = np.zeros((n, n), dtype=int)
        degrees = np.zeros(n, dtype=int)
        
        # Create initial clique
        for i in range(m):
            for j in range(i + 1, m):
                adjacency[i, j] = 1
                adjacency[j, i] = 1
                degrees[i] += 1
                degrees[j] += 1
        
        # Preferential attachment
        for new_node in range(m, n):
            # Calculate attachment probabilities
            total_degree = np.sum(degrees[:new_node])
            if total_degree > 0:
                probabilities = degrees[:new_node] / total_degree
            else:
                probabilities = np.ones(new_node) / new_node
            
            # Select m nodes to connect to
            selected = np.random.choice(
                new_node,
                size=min(m, new_node),
                replace=False,
                p=probabilities
            )
            
            # Add edges
            for node in selected:
                adjacency[new_node, node] = 1
                adjacency[node, new_node] = 1
                degrees[new_node] += 1
                degrees[node] += 1
        
        self.adjacency_matrix = adjacency
        self.degrees = degrees
        
        # Build edge list
        self.edges = []
        for i in range(n):
            for j in range(i + 1, n):
                if adjacency[i, j] == 1:
                    self.edges.append((i, j))
        
        return adjacency
    
    def get_degree_distribution(self) -> Dict[int, int]:
        """Get degree distribution"""
        if self.degrees is None:
            return {}
        
        distribution = defaultdict(int)
        for degree in self.degrees:
            distribution[int(degree)] += 1
        
        return dict(distribution)


class NetworkCentrality:
    """
    Network Centrality Measures
    
    Identify important nodes in networks
    """
    
    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Initialize centrality calculator
        
        Args:
            adjacency_matrix: Network adjacency matrix
        """
        self.adjacency_matrix = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
    
    def degree_centrality(self) -> np.ndarray:
        """Degree centrality: number of connections"""
        return np.sum(self.adjacency_matrix, axis=1) / (self.n_nodes - 1)
    
    def betweenness_centrality(self) -> np.ndarray:
        """Betweenness centrality: fraction of shortest paths passing through node"""
        centrality = np.zeros(self.n_nodes)
        
        for s in range(self.n_nodes):
            # BFS to find shortest paths
            distances = np.full(self.n_nodes, -1)
            paths = defaultdict(list)
            queue = deque([s])
            distances[s] = 0
            paths[s] = [[s]]
            
            while queue:
                v = queue.popleft()
                for w in range(self.n_nodes):
                    if self.adjacency_matrix[v, w] == 1:
                        if distances[w] == -1:
                            distances[w] = distances[v] + 1
                            queue.append(w)
                            paths[w] = [p + [w] for p in paths[v]]
                        elif distances[w] == distances[v] + 1:
                            paths[w].extend([p + [w] for p in paths[v]])
            
            # Count paths through each node
            for t in range(self.n_nodes):
                if t != s and len(paths[t]) > 0:
                    for path in paths[t]:
                        for node in path[1:-1]:  # Exclude endpoints
                            centrality[node] += 1.0 / len(paths[t])
        
        # Normalize
        if self.n_nodes > 2:
            centrality /= ((self.n_nodes - 1) * (self.n_nodes - 2))
        
        return centrality
    
    def closeness_centrality(self) -> np.ndarray:
        """Closeness centrality: inverse of average distance to all nodes"""
        centrality = np.zeros(self.n_nodes)
        
        for node in range(self.n_nodes):
            # BFS to find distances
            distances = np.full(self.n_nodes, -1)
            queue = deque([node])
            distances[node] = 0
            
            while queue:
                v = queue.popleft()
                for w in range(self.n_nodes):
                    if self.adjacency_matrix[v, w] == 1 and distances[w] == -1:
                        distances[w] = distances[v] + 1
                        queue.append(w)
            
            # Calculate average distance
            reachable = distances[distances >= 0]
            if len(reachable) > 1:
                avg_distance = np.mean(reachable[reachable > 0])
                if avg_distance > 0:
                    centrality[node] = 1.0 / avg_distance
        
        return centrality
    
    def eigenvector_centrality(self, max_iter: int = 100, tol: float = 1e-6) -> np.ndarray:
        """Eigenvector centrality: importance based on neighbors' importance"""
        centrality = np.ones(self.n_nodes) / self.n_nodes
        
        for _ in range(max_iter):
            new_centrality = np.dot(self.adjacency_matrix, centrality)
            norm = np.linalg.norm(new_centrality)
            if norm == 0:
                break
            new_centrality /= norm
            
            if np.linalg.norm(new_centrality - centrality) < tol:
                break
            centrality = new_centrality
        
        return centrality


class CommunityDetection:
    """
    Community Detection in Networks
    
    Identify clusters/communities in networks
    """
    
    def __init__(self, adjacency_matrix: np.ndarray):
        """
        Initialize community detection
        
        Args:
            adjacency_matrix: Network adjacency matrix
        """
        self.adjacency_matrix = adjacency_matrix
        self.n_nodes = adjacency_matrix.shape[0]
    
    def modularity(self, communities: List[Set[int]]) -> float:
        """
        Calculate modularity of community structure
        
        Args:
            communities: List of sets of node indices
        
        Returns:
            Modularity score
        """
        m = np.sum(self.adjacency_matrix) / 2  # Total edges
        if m == 0:
            return 0.0
        
        modularity = 0.0
        for community in communities:
            for i in community:
                for j in community:
                    if i != j:
                        A_ij = self.adjacency_matrix[i, j]
                        k_i = np.sum(self.adjacency_matrix[i])
                        k_j = np.sum(self.adjacency_matrix[j])
                        modularity += A_ij - (k_i * k_j) / (2 * m)
        
        return modularity / (2 * m)
    
    def greedy_modularity(self) -> List[Set[int]]:
        """
        Greedy modularity maximization for community detection
        
        Returns:
            List of communities (sets of node indices)
        """
        # Initialize: each node is its own community
        communities = [{i} for i in range(self.n_nodes)]
        
        improved = True
        while improved:
            improved = False
            best_modularity = self.modularity(communities)
            
            # Try merging communities
            for i in range(len(communities)):
                for j in range(i + 1, len(communities)):
                    # Merge communities i and j
                    merged = communities[i] | communities[j]
                    new_communities = communities.copy()
                    new_communities[i] = merged
                    new_communities.pop(j)
                    
                    new_modularity = self.modularity(new_communities)
                    
                    if new_modularity > best_modularity:
                        communities = new_communities
                        best_modularity = new_modularity
                        improved = True
                        break
                
                if improved:
                    break
        
        return communities


def network_based_feature_importance(
    X: np.ndarray,
    y: np.ndarray,
    method: str = 'betweenness'
) -> np.ndarray:
    """
    Calculate feature importance using network centrality
    
    Args:
        X: Feature matrix
        y: Target vector
        method: 'degree', 'betweenness', 'closeness', 'eigenvector'
    
    Returns:
        Feature importance scores
    """
    from sklearn.metrics.pairwise import cosine_similarity
    
    # Build feature similarity network
    # Features are nodes, edges based on correlation/similarity
    similarity_matrix = np.abs(cosine_similarity(X.T))
    
    # Threshold to create adjacency matrix
    threshold = np.percentile(similarity_matrix, 75)
    adjacency = (similarity_matrix > threshold).astype(int)
    np.fill_diagonal(adjacency, 0)
    
    # Calculate centrality
    centrality_calc = NetworkCentrality(adjacency)
    
    if method == 'degree':
        importance = centrality_calc.degree_centrality()
    elif method == 'betweenness':
        importance = centrality_calc.betweenness_centrality()
    elif method == 'closeness':
        importance = centrality_calc.closeness_centrality()
    elif method == 'eigenvector':
        importance = centrality_calc.eigenvector_centrality()
    else:
        raise ValueError(f"Unknown method: {method}")
    
    return importance
