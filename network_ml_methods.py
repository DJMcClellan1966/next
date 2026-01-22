"""
Network ML Methods - ML-Relevant Network Methods Inspired by Stevens TCP/IP
Focuses on network graph analysis, distributed ML, and network optimization for ML workflows

Inspired by:
- W. Richard Stevens "TCP/IP Illustrated" - Network patterns and optimization
- Network graph analysis for ML
- Distributed ML training patterns
- Network protocol optimization for ML serving
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Union, Callable
import numpy as np
from collections import defaultdict, deque
import heapq
import math
import time
from threading import Lock

sys.path.insert(0, str(Path(__file__).parent))


class NetworkGraphAnalysis:
    """
    Network Graph Analysis for ML
    
    Analyze network topologies, connection patterns, and extract ML features
    """
    
    def __init__(self):
        """Initialize network graph analyzer"""
        self.graph = defaultdict(list)
        self.node_features = {}
        self.edge_features = {}
    
    def build_graph_from_connections(
        self,
        connections: List[Tuple[Any, Any, Optional[Dict]]]
    ):
        """
        Build network graph from connections
        
        Args:
            connections: List of (source, target, edge_attrs) tuples
        """
        self.graph = defaultdict(list)
        self.edge_features = {}
        
        for conn in connections:
            if len(conn) == 1:
                # Isolated node (no connections)
                source = conn[0]
                self.graph[source] = []  # Initialize empty
                continue
            elif len(conn) == 2:
                source, target = conn
                attrs = {}
            else:
                source, target, attrs = conn
            
            self.graph[source].append(target)
            self.edge_features[(source, target)] = attrs
    
    def extract_topology_features(self) -> Dict[str, Any]:
        """
        Extract network topology features for ML
        
        Returns:
            Dictionary of topology features
        """
        if not self.graph:
            return {}
        
        nodes = set(self.graph.keys())
        for neighbors in self.graph.values():
            nodes.update(neighbors)
        
        # Basic statistics
        n_nodes = len(nodes)
        n_edges = sum(len(neighbors) for neighbors in self.graph.values())
        
        # Degree statistics
        degrees = [len(self.graph.get(node, [])) for node in nodes]
        in_degrees = defaultdict(int)
        for neighbors in self.graph.values():
            for neighbor in neighbors:
                in_degrees[neighbor] += 1
        
        all_degrees = degrees + list(in_degrees.values())
        
        # Centrality measures (simplified)
        degree_centrality = {node: deg / (n_nodes - 1) if n_nodes > 1 else 0 
                            for node, deg in zip(nodes, degrees)}
        
        # Clustering coefficient (simplified)
        clustering = {}
        for node in nodes:
            neighbors = set(self.graph.get(node, []))
            if len(neighbors) < 2:
                clustering[node] = 0.0
            else:
                edges_between = sum(
                    1 for n1 in neighbors
                    for n2 in neighbors
                    if n1 in self.graph.get(n2, [])
                )
                possible = len(neighbors) * (len(neighbors) - 1)
                clustering[node] = edges_between / possible if possible > 0 else 0.0
        
        return {
            'n_nodes': n_nodes,
            'n_edges': n_edges,
            'density': (2 * n_edges) / (n_nodes * (n_nodes - 1)) if n_nodes > 1 else 0,
            'avg_degree': np.mean(all_degrees) if all_degrees else 0,
            'max_degree': max(all_degrees) if all_degrees else 0,
            'min_degree': min(all_degrees) if all_degrees else 0,
            'degree_centrality': degree_centrality,
            'clustering_coefficient': np.mean(list(clustering.values())) if clustering else 0,
            'node_clustering': clustering
        }
    
    def detect_connection_patterns(self) -> Dict[str, List]:
        """
        Detect connection patterns in network
        
        Returns:
            Dictionary of detected patterns
        """
        patterns = {
            'hubs': [],  # High-degree nodes
            'isolates': [],  # Isolated nodes
            'cliques': [],  # Fully connected subgraphs
            'chains': []  # Linear chains
        }
        
        if not self.graph:
            return patterns
        
        nodes = set(self.graph.keys())
        for neighbors in self.graph.values():
            nodes.update(neighbors)
        
        # Find hubs (high degree)
        degrees = {node: len(self.graph.get(node, [])) for node in nodes}
        avg_degree = np.mean(list(degrees.values())) if degrees else 0
        patterns['hubs'] = [node for node, deg in degrees.items() 
                           if deg > 2 * avg_degree]
        
        # Find isolates
        patterns['isolates'] = [node for node in nodes 
                               if degrees.get(node, 0) == 0 and 
                               not any(node in self.graph[n] for n in self.graph)]
        
        # Find chains (simplified - nodes with degree 2)
        patterns['chains'] = [node for node, deg in degrees.items() if deg == 2]
        
        return patterns
    
    def extract_node_features(self, node: Any) -> Dict[str, float]:
        """
        Extract features for a specific node
        
        Args:
            node: Node identifier
            
        Returns:
            Dictionary of node features
        """
        neighbors = self.graph.get(node, [])
        in_degree = sum(1 for neighbors_list in self.graph.values() 
                       for n in neighbors_list if n == node)
        out_degree = len(neighbors)
        
        # Edge features
        edge_weights = []
        for neighbor in neighbors:
            edge_attrs = self.edge_features.get((node, neighbor), {})
            if 'weight' in edge_attrs:
                edge_weights.append(edge_attrs['weight'])
        
        return {
            'in_degree': in_degree,
            'out_degree': out_degree,
            'total_degree': in_degree + out_degree,
            'avg_edge_weight': np.mean(edge_weights) if edge_weights else 0.0,
            'max_edge_weight': max(edge_weights) if edge_weights else 0.0,
            'min_edge_weight': min(edge_weights) if edge_weights else 0.0,
            'n_neighbors': len(neighbors)
        }
    
    def prepare_for_gnn(self) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Prepare network data for Graph Neural Network
        
        Returns:
            Tuple (node_features, edge_index, edge_weights)
        """
        if not self.graph:
            return np.array([]), np.array([]), np.array([])
        
        nodes = sorted(set(self.graph.keys()) | 
                     set(n for neighbors in self.graph.values() for n in neighbors))
        node_to_idx = {node: i for i, node in enumerate(nodes)}
        
        # Node features
        node_features = np.array([
            list(self.extract_node_features(node).values())
            for node in nodes
        ])
        
        # Edge index
        edge_index = []
        edge_weights = []
        
        for source in self.graph:
            for target in self.graph[source]:
                edge_index.append([node_to_idx[source], node_to_idx[target]])
                edge_attrs = self.edge_features.get((source, target), {})
                weight = edge_attrs.get('weight', 1.0)
                edge_weights.append(weight)
        
        return (
            node_features,
            np.array(edge_index).T if edge_index else np.array([]),
            np.array(edge_weights) if edge_weights else np.array([])
        )


class DistributedMLPatterns:
    """
    Distributed ML Patterns Inspired by Network Communication
    
    Parameter server, federated learning, model synchronization
    """
    
    class ParameterServer:
        """Parameter Server for Distributed Training"""
        
        def __init__(self, initial_params: Dict[str, np.ndarray]):
            """
            Args:
                initial_params: Initial model parameters
            """
            self.params = {k: v.copy() for k, v in initial_params.items()}
            self.lock = Lock()
            self.update_count = 0
        
        def get_params(self) -> Dict[str, np.ndarray]:
            """Get current parameters"""
            with self.lock:
                return {k: v.copy() for k, v in self.params.items()}
        
        def update_params(
            self,
            updates: Dict[str, np.ndarray],
            learning_rate: float = 1.0
        ):
            """
            Update parameters with gradients
            
            Args:
                updates: Parameter updates (gradients)
                learning_rate: Learning rate
            """
            with self.lock:
                for key, update in updates.items():
                    if key in self.params:
                        self.params[key] -= learning_rate * update
                self.update_count += 1
        
        def get_update_count(self) -> int:
            """Get number of updates"""
            return self.update_count
    
    @staticmethod
    def federated_learning_round(
        clients: List[Callable],
        server_params: Dict[str, np.ndarray],
        aggregation: str = 'average'
    ) -> Dict[str, np.ndarray]:
        """
        Federated Learning Round
        
        Args:
            clients: List of client update functions (return gradients)
            server_params: Server model parameters
            aggregation: Aggregation method ('average', 'weighted')
            
        Returns:
            Updated server parameters
        """
        all_updates = []
        
        for client_func in clients:
            updates = client_func(server_params)
            all_updates.append(updates)
        
        # Aggregate updates
        aggregated = {}
        for key in server_params.keys():
            if aggregation == 'average':
                aggregated[key] = np.mean(
                    [updates.get(key, np.zeros_like(server_params[key])) 
                     for updates in all_updates],
                    axis=0
                )
            elif aggregation == 'weighted':
                # Weighted average (simplified - equal weights)
                weights = [1.0 / len(all_updates)] * len(all_updates)
                aggregated[key] = np.average(
                    [updates.get(key, np.zeros_like(server_params[key])) 
                     for updates in all_updates],
                    axis=0,
                    weights=weights
                )
        
        # Update server parameters
        updated = {}
        for key in server_params.keys():
            updated[key] = server_params[key] - aggregated[key]
        
        return updated
    
    @staticmethod
    def model_synchronization(
        models: List[Dict[str, np.ndarray]],
        method: str = 'average'
    ) -> Dict[str, np.ndarray]:
        """
        Synchronize multiple model copies
        
        Args:
            models: List of model parameter dictionaries
            method: Synchronization method ('average', 'majority')
            
        Returns:
            Synchronized model parameters
        """
        if not models:
            return {}
        
        if method == 'average':
            synchronized = {}
            for key in models[0].keys():
                synchronized[key] = np.mean(
                    [model[key] for model in models],
                    axis=0
                )
            return synchronized
        
        elif method == 'majority':
            # Simplified: use average (true majority voting would need voting mechanism)
            return DistributedMLPatterns.model_synchronization(models, 'average')
        
        return models[0] if models else {}


class NetworkOptimization:
    """
    Network Protocol Optimization for ML Serving
    
    Connection pooling, protocol-level caching, load balancing
    """
    
    class ConnectionPool:
        """Connection Pool for Efficient Resource Usage"""
        
        def __init__(self, max_size: int = 10, timeout: float = 30.0):
            """
            Args:
                max_size: Maximum pool size
                timeout: Connection timeout
            """
            self.max_size = max_size
            self.timeout = timeout
            self.pool = deque()
            self.active = set()
            self.lock = Lock()
            self.created = 0
        
        def acquire(self) -> Optional[Any]:
            """Acquire connection from pool"""
            with self.lock:
                if self.pool:
                    conn = self.pool.popleft()
                    self.active.add(conn)
                    return conn
                
                if len(self.active) < self.max_size:
                    conn = self._create_connection()
                    self.active.add(conn)
                    self.created += 1
                    return conn
            
            return None
        
        def release(self, conn: Any):
            """Release connection back to pool"""
            with self.lock:
                if conn in self.active:
                    self.active.remove(conn)
                    self.pool.append(conn)
        
        def _create_connection(self) -> Any:
            """Create new connection (placeholder)"""
            return f"connection_{self.created}"
        
        def get_stats(self) -> Dict[str, Any]:
            """Get pool statistics"""
            with self.lock:
                return {
                    'pool_size': len(self.pool),
                    'active_count': len(self.active),
                    'total_created': self.created,
                    'max_size': self.max_size
                }
    
    class ProtocolCache:
        """Protocol-Level Caching for Network Performance"""
        
        def __init__(self, max_size: int = 1000, ttl: float = 3600.0):
            """
            Args:
                max_size: Maximum cache size
                ttl: Time-to-live in seconds
            """
            self.max_size = max_size
            self.ttl = ttl
            self.cache = {}
            self.timestamps = {}
            self.lock = Lock()
        
        def get(self, key: str) -> Optional[Any]:
            """Get cached value"""
            with self.lock:
                if key in self.cache:
                    if time.time() - self.timestamps[key] < self.ttl:
                        return self.cache[key]
                    else:
                        # Expired
                        del self.cache[key]
                        del self.timestamps[key]
            return None
        
        def set(self, key: str, value: Any):
            """Set cached value"""
            with self.lock:
                if len(self.cache) >= self.max_size:
                    # Evict oldest
                    oldest_key = min(self.timestamps.keys(), 
                                    key=lambda k: self.timestamps[k])
                    del self.cache[oldest_key]
                    del self.timestamps[oldest_key]
                
                self.cache[key] = value
                self.timestamps[key] = time.time()
        
        def clear(self):
            """Clear cache"""
            with self.lock:
                self.cache.clear()
                self.timestamps.clear()
        
        def get_stats(self) -> Dict[str, Any]:
            """Get cache statistics"""
            with self.lock:
                return {
                    'size': len(self.cache),
                    'max_size': self.max_size,
                    'hit_rate': 0.0  # Would track in production
                }
    
    @staticmethod
    def load_balance_requests(
        servers: List[Any],
        method: str = 'round_robin'
    ) -> Callable:
        """
        Create load balancer function
        
        Args:
            servers: List of server identifiers
            method: Balancing method ('round_robin', 'random', 'least_connections')
            
        Returns:
            Load balancer function
        """
        if method == 'round_robin':
            index = 0
            def balancer():
                nonlocal index
                server = servers[index % len(servers)]
                index += 1
                return server
            return balancer
        
        elif method == 'random':
            def balancer():
                return np.random.choice(servers)
            return balancer
        
        elif method == 'least_connections':
            connections = {server: 0 for server in servers}
            def balancer():
                server = min(connections.keys(), key=lambda s: connections[s])
                connections[server] += 1
                return server
            return balancer
        
        # Default: round robin
        return NetworkOptimization.load_balance_requests(servers, 'round_robin')


class NetworkMLMethods:
    """
    Unified interface for all network ML methods
    """
    
    def __init__(self):
        self.graph_analysis = NetworkGraphAnalysis()
        self.distributed_ml = DistributedMLPatterns()
        self.optimization = NetworkOptimization()
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'numpy': 'numpy>=1.26.0',
            'python': 'Python 3.8+'
        }
