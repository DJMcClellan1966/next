"""
Tests for Network ML Methods
Test network graph analysis, distributed ML, and network optimization
"""
import sys
from pathlib import Path
import numpy as np
import pytest

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
    pytestmark = pytest.mark.skip("Network ML methods not available")


class TestNetworkGraphAnalysis:
    """Tests for network graph analysis"""
    
    def test_build_graph(self):
        """Test graph building"""
        analyzer = NetworkGraphAnalysis()
        connections = [
            ('A', 'B'),
            ('B', 'C'),
            ('A', 'C')
        ]
        analyzer.build_graph_from_connections(connections)
        
        assert 'A' in analyzer.graph
        assert 'B' in analyzer.graph['A']
    
    def test_extract_topology_features(self):
        """Test topology feature extraction"""
        analyzer = NetworkGraphAnalysis()
        connections = [
            ('A', 'B'),
            ('B', 'C'),
            ('C', 'A')
        ]
        analyzer.build_graph_from_connections(connections)
        
        features = analyzer.extract_topology_features()
        assert 'n_nodes' in features
        assert features['n_nodes'] == 3
        assert features['n_edges'] == 3
    
    def test_detect_connection_patterns(self):
        """Test connection pattern detection"""
        analyzer = NetworkGraphAnalysis()
        connections = [
            ('A', 'B'),
            ('A', 'C'),
            ('A', 'D'),  # A is a hub
            ('E', 'F'),
            ('G',)  # Isolate
        ]
        analyzer.build_graph_from_connections(connections)
        
        patterns = analyzer.detect_connection_patterns()
        assert 'hubs' in patterns
        assert 'isolates' in patterns
    
    def test_extract_node_features(self):
        """Test node feature extraction"""
        analyzer = NetworkGraphAnalysis()
        connections = [
            ('A', 'B', {'weight': 1.0}),
            ('A', 'C', {'weight': 2.0})
        ]
        analyzer.build_graph_from_connections(connections)
        
        features = analyzer.extract_node_features('A')
        assert 'out_degree' in features
        assert features['out_degree'] == 2
    
    def test_prepare_for_gnn(self):
        """Test GNN data preparation"""
        analyzer = NetworkGraphAnalysis()
        connections = [
            ('A', 'B'),
            ('B', 'C')
        ]
        analyzer.build_graph_from_connections(connections)
        
        node_features, edge_index, edge_weights = analyzer.prepare_for_gnn()
        assert node_features.shape[0] > 0
        assert edge_index.shape[0] == 2  # Source and target


class TestDistributedMLPatterns:
    """Tests for distributed ML patterns"""
    
    def test_parameter_server(self):
        """Test parameter server"""
        initial_params = {
            'weight1': np.array([1.0, 2.0]),
            'weight2': np.array([3.0, 4.0])
        }
        ps = DistributedMLPatterns.ParameterServer(initial_params)
        
        params = ps.get_params()
        assert 'weight1' in params
        
        updates = {
            'weight1': np.array([0.1, 0.1])
        }
        ps.update_params(updates, learning_rate=0.1)
        
        updated = ps.get_params()
        assert updated['weight1'][0] < params['weight1'][0]
    
    def test_federated_learning_round(self):
        """Test federated learning"""
        def client1(params):
            return {'weight': np.array([0.1])}
        
        def client2(params):
            return {'weight': np.array([0.2])}
        
        server_params = {'weight': np.array([1.0])}
        clients = [client1, client2]
        
        updated = DistributedMLPatterns.federated_learning_round(
            clients, server_params
        )
        assert 'weight' in updated
    
    def test_model_synchronization(self):
        """Test model synchronization"""
        models = [
            {'weight': np.array([1.0, 2.0])},
            {'weight': np.array([2.0, 3.0])},
            {'weight': np.array([3.0, 4.0])}
        ]
        
        synced = DistributedMLPatterns.model_synchronization(models)
        assert 'weight' in synced
        assert synced['weight'][0] == 2.0  # Average


class TestNetworkOptimization:
    """Tests for network optimization"""
    
    def test_connection_pool(self):
        """Test connection pool"""
        pool = NetworkOptimization.ConnectionPool(max_size=5)
        
        conn1 = pool.acquire()
        assert conn1 is not None
        
        pool.release(conn1)
        
        stats = pool.get_stats()
        assert 'pool_size' in stats
    
    def test_protocol_cache(self):
        """Test protocol cache"""
        cache = NetworkOptimization.ProtocolCache(max_size=10)
        
        cache.set('key1', 'value1')
        value = cache.get('key1')
        assert value == 'value1'
        
        assert cache.get('key2') is None
        
        stats = cache.get_stats()
        assert 'size' in stats
    
    def test_load_balance_requests(self):
        """Test load balancing"""
        servers = ['server1', 'server2', 'server3']
        balancer = NetworkOptimization.load_balance_requests(servers, 'round_robin')
        
        server1 = balancer()
        server2 = balancer()
        
        assert server1 in servers
        assert server2 in servers
        assert server1 != server2 or len(servers) == 1


class TestNetworkMLMethods:
    """Test unified interface"""
    
    def test_unified_interface(self):
        """Test NetworkMLMethods"""
        network_ml = NetworkMLMethods()
        
        assert network_ml.graph_analysis is not None
        assert network_ml.distributed_ml is not None
        assert network_ml.optimization is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
