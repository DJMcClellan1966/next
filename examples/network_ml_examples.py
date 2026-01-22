"""
Network ML Methods - Examples
Demonstrates network graph analysis, distributed ML, and network optimization
"""
import sys
from pathlib import Path
import numpy as np

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from network_ml_methods import (
        NetworkGraphAnalysis,
        DistributedMLPatterns,
        NetworkOptimization,
        NetworkMLMethods
    )
    NETWORK_ML_AVAILABLE = True
except ImportError:
    NETWORK_ML_AVAILABLE = False
    print("Warning: Network ML methods not available")
    exit(1)


def example_network_graph_analysis():
    """Example: Network graph analysis for ML"""
    print("=" * 60)
    print("Example 1: Network Graph Analysis for ML")
    print("=" * 60)
    
    # Build network graph (e.g., social network, communication network)
    analyzer = NetworkGraphAnalysis()
    connections = [
        ('Alice', 'Bob', {'weight': 1.0}),
        ('Bob', 'Charlie', {'weight': 2.0}),
        ('Alice', 'Charlie', {'weight': 1.5}),
        ('Charlie', 'David', {'weight': 1.0}),
        ('David', 'Eve', {'weight': 1.0})
    ]
    analyzer.build_graph_from_connections(connections)
    
    # Extract topology features
    print("\nTopology Features:")
    features = analyzer.extract_topology_features()
    print(f"Nodes: {features['n_nodes']}")
    print(f"Edges: {features['n_edges']}")
    print(f"Density: {features['density']:.3f}")
    print(f"Average Degree: {features['avg_degree']:.2f}")
    print(f"Clustering Coefficient: {features['clustering_coefficient']:.3f}")
    
    # Detect patterns
    print("\nConnection Patterns:")
    patterns = analyzer.detect_connection_patterns()
    print(f"Hubs: {patterns['hubs']}")
    print(f"Isolates: {patterns['isolates']}")
    
    # Extract node features
    print("\nNode Features for 'Alice':")
    node_features = analyzer.extract_node_features('Alice')
    for key, value in node_features.items():
        print(f"  {key}: {value}")
    
    # Prepare for GNN
    print("\nPreparing for Graph Neural Network...")
    node_features, edge_index, edge_weights = analyzer.prepare_for_gnn()
    print(f"Node features shape: {node_features.shape}")
    print(f"Edge index shape: {edge_index.shape}")
    print(f"Edge weights: {edge_weights}")
    
    print("\n✓ Network graph analysis example complete\n")


def example_distributed_ml():
    """Example: Distributed ML patterns"""
    print("=" * 60)
    print("Example 2: Distributed ML Patterns")
    print("=" * 60)
    
    # Parameter Server
    print("\nParameter Server:")
    initial_params = {
        'weights': np.array([1.0, 2.0, 3.0]),
        'bias': np.array([0.5])
    }
    ps = DistributedMLPatterns.ParameterServer(initial_params)
    
    print(f"Initial params: {ps.get_params()}")
    
    # Simulate worker updates
    updates = {
        'weights': np.array([0.1, 0.2, 0.3])
    }
    ps.update_params(updates, learning_rate=0.1)
    
    print(f"After update: {ps.get_params()}")
    print(f"Update count: {ps.get_update_count()}")
    
    # Federated Learning
    print("\nFederated Learning Round:")
    def client1(params):
        return {'weights': np.array([0.1, 0.1, 0.1])}
    
    def client2(params):
        return {'weights': np.array([0.2, 0.2, 0.2])}
    
    server_params = {'weights': np.array([1.0, 1.0, 1.0])}
    clients = [client1, client2]
    
    updated = DistributedMLPatterns.federated_learning_round(
        clients, server_params, aggregation='average'
    )
    print(f"Server params after federated round: {updated}")
    
    # Model Synchronization
    print("\nModel Synchronization:")
    models = [
        {'weights': np.array([1.0, 2.0])},
        {'weights': np.array([2.0, 3.0])},
        {'weights': np.array([3.0, 4.0])}
    ]
    synced = DistributedMLPatterns.model_synchronization(models, method='average')
    print(f"Synchronized weights: {synced['weights']}")
    
    print("\n✓ Distributed ML example complete\n")


def example_network_optimization():
    """Example: Network optimization for ML serving"""
    print("=" * 60)
    print("Example 3: Network Optimization for ML Serving")
    print("=" * 60)
    
    # Connection Pool
    print("\nConnection Pool:")
    pool = NetworkOptimization.ConnectionPool(max_size=5)
    
    conn1 = pool.acquire()
    conn2 = pool.acquire()
    print(f"Acquired connections: {conn1}, {conn2}")
    
    pool.release(conn1)
    stats = pool.get_stats()
    print(f"Pool stats: {stats}")
    
    # Protocol Cache
    print("\nProtocol Cache:")
    cache = NetworkOptimization.ProtocolCache(max_size=100, ttl=3600)
    
    cache.set('model_prediction_123', {'prediction': [0.8, 0.2]})
    cache.set('model_prediction_456', {'prediction': [0.6, 0.4]})
    
    cached = cache.get('model_prediction_123')
    print(f"Cached value: {cached}")
    
    stats = cache.get_stats()
    print(f"Cache stats: {stats}")
    
    # Load Balancing
    print("\nLoad Balancing:")
    servers = ['model_server_1', 'model_server_2', 'model_server_3']
    balancer = NetworkOptimization.load_balance_requests(servers, 'round_robin')
    
    print("Request routing (round-robin):")
    for i in range(5):
        server = balancer()
        print(f"  Request {i+1} -> {server}")
    
    print("\n✓ Network optimization example complete\n")


def example_integrated_workflow():
    """Example: Integrated network ML workflow"""
    print("=" * 60)
    print("Example 4: Integrated Network ML Workflow")
    print("=" * 60)
    
    network_ml = NetworkMLMethods()
    
    # Step 1: Analyze network graph
    print("\nStep 1: Network Graph Analysis...")
    connections = [
        ('user1', 'user2'),
        ('user2', 'user3'),
        ('user1', 'user3')
    ]
    network_ml.graph_analysis.build_graph_from_connections(connections)
    features = network_ml.graph_analysis.extract_topology_features()
    print(f"Network has {features['n_nodes']} nodes, {features['n_edges']} edges")
    
    # Step 2: Distributed training
    print("\nStep 2: Distributed Training...")
    ps = network_ml.distributed_ml.ParameterServer({
        'weights': np.array([1.0, 2.0])
    })
    ps.update_params({'weights': np.array([0.1, 0.1])}, learning_rate=0.1)
    print(f"Parameters updated: {ps.get_params()}")
    
    # Step 3: Network optimization
    print("\nStep 3: Network Optimization...")
    pool = network_ml.optimization.ConnectionPool(max_size=10)
    conn = pool.acquire()
    print(f"Connection acquired: {conn}")
    pool.release(conn)
    
    print("\n✓ Integrated workflow example complete\n")


if __name__ == '__main__':
    print("\n" + "=" * 60)
    print("Network ML Methods - Examples")
    print("=" * 60 + "\n")
    
    if not NETWORK_ML_AVAILABLE:
        print("Required dependencies not available")
        exit(1)
    
    try:
        example_network_graph_analysis()
        example_distributed_ml()
        example_network_optimization()
        example_integrated_workflow()
        
        print("=" * 60)
        print("All network ML examples completed successfully!")
        print("=" * 60)
    except Exception as e:
        print(f"\nError running examples: {e}")
        import traceback
        traceback.print_exc()
