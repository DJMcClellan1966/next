"""
Knowledge Graph Agent - Build and query knowledge graphs

Implements:
- Knowledge graph construction
- Entity extraction
- Relationship mapping
- Graph queries
- Semantic reasoning
"""
from typing import Dict, List, Optional, Any, Set, Tuple
import logging
from collections import defaultdict

logger = logging.getLogger(__name__)


class KnowledgeGraph:
    """
    Knowledge Graph - Structured knowledge representation
    
    Nodes: Entities (concepts, objects, people, etc.)
    Edges: Relationships (is-a, part-of, related-to, etc.)
    """
    
    def __init__(self):
        self.nodes = {}  # node_id -> {type, properties, embeddings}
        self.edges = defaultdict(list)  # source_id -> [(target_id, relationship, properties)]
        self.node_index = {}  # node_name -> node_id
        self.relationship_types = set()
    
    def add_node(self, node_id: str, node_type: str, properties: Optional[Dict] = None, 
                embedding: Optional[Any] = None):
        """
        Add node to knowledge graph
        
        Parameters
        ----------
        node_id : str
            Unique node identifier
        node_type : str
            Type of node (entity, concept, etc.)
        properties : dict, optional
            Node properties
        embedding : array-like, optional
            Node embedding for semantic search
        """
        self.nodes[node_id] = {
            'type': node_type,
            'properties': properties or {},
            'embedding': embedding
        }
        
        # Index by name if available
        if properties and 'name' in properties:
            self.node_index[properties['name']] = node_id
    
    def add_edge(self, source_id: str, target_id: str, relationship: str, 
                properties: Optional[Dict] = None):
        """
        Add edge (relationship) between nodes
        
        Parameters
        ----------
        source_id : str
            Source node ID
        target_id : str
            Target node ID
        relationship : str
            Type of relationship
        properties : dict, optional
            Edge properties
        """
        if source_id not in self.nodes or target_id not in self.nodes:
            logger.warning(f"Cannot add edge: nodes not found")
            return
        
        self.edges[source_id].append({
            'target': target_id,
            'relationship': relationship,
            'properties': properties or {}
        })
        
        self.relationship_types.add(relationship)
    
    def get_node(self, node_id: str) -> Optional[Dict]:
        """Get node by ID"""
        return self.nodes.get(node_id)
    
    def get_neighbors(self, node_id: str, relationship: Optional[str] = None) -> List[Dict]:
        """
        Get neighboring nodes
        
        Parameters
        ----------
        node_id : str
            Node ID
        relationship : str, optional
            Filter by relationship type
            
        Returns
        -------
        neighbors : list of dict
            Neighboring nodes with relationships
        """
        if node_id not in self.edges:
            return []
        
        neighbors = []
        for edge in self.edges[node_id]:
            if relationship is None or edge['relationship'] == relationship:
                neighbor_node = self.get_node(edge['target'])
                if neighbor_node:
                    neighbors.append({
                        'node': neighbor_node,
                        'node_id': edge['target'],
                        'relationship': edge['relationship'],
                        'properties': edge['properties']
                    })
        
        return neighbors
    
    def find_path(self, source_id: str, target_id: str, max_depth: int = 3) -> List[List[str]]:
        """
        Find paths between nodes
        
        Parameters
        ----------
        source_id : str
            Source node ID
        target_id : str
            Target node ID
        max_depth : int
            Maximum path depth
            
        Returns
        -------
        paths : list of lists
            List of paths (each path is a list of node IDs)
        """
        paths = []
        
        def dfs(current: str, target: str, path: List[str], depth: int):
            if depth > max_depth:
                return
            
            if current == target:
                paths.append(path.copy())
                return
            
            if current not in self.edges:
                return
            
            for edge in self.edges[current]:
                next_node = edge['target']
                if next_node not in path:  # Avoid cycles
                    path.append(next_node)
                    dfs(next_node, target, path, depth + 1)
                    path.pop()
        
        dfs(source_id, target_id, [source_id], 0)
        return paths
    
    def query(self, query_type: str, **kwargs) -> List[Dict]:
        """
        Query knowledge graph
        
        Parameters
        ----------
        query_type : str
            Type of query ('find_node', 'find_relationship', 'find_path', 'find_by_type')
        **kwargs
            Query parameters
            
        Returns
        -------
        results : list of dict
            Query results
        """
        if query_type == 'find_node':
            name = kwargs.get('name')
            if name and name in self.node_index:
                node_id = self.node_index[name]
                return [{'node_id': node_id, 'node': self.get_node(node_id)}]
            return []
        
        elif query_type == 'find_by_type':
            node_type = kwargs.get('type')
            results = []
            for node_id, node in self.nodes.items():
                if node['type'] == node_type:
                    results.append({'node_id': node_id, 'node': node})
            return results
        
        elif query_type == 'find_relationship':
            source_id = kwargs.get('source_id')
            relationship = kwargs.get('relationship')
            if source_id:
                return self.get_neighbors(source_id, relationship)
            return []
        
        elif query_type == 'find_path':
            source_id = kwargs.get('source_id')
            target_id = kwargs.get('target_id')
            max_depth = kwargs.get('max_depth', 3)
            if source_id and target_id:
                paths = self.find_path(source_id, target_id, max_depth)
                return [{'path': path} for path in paths]
            return []
        
        return []
    
    def get_statistics(self) -> Dict:
        """Get knowledge graph statistics"""
        return {
            'num_nodes': len(self.nodes),
            'num_edges': sum(len(edges) for edges in self.edges.values()),
            'num_relationship_types': len(self.relationship_types),
            'relationship_types': list(self.relationship_types)
        }


class KnowledgeGraphAgent:
    """
    Knowledge Graph Agent
    
    Builds and queries knowledge graphs for structured knowledge
    """
    
    def __init__(self):
        self.graph = KnowledgeGraph()
        self.entity_extractors = {}
        self.relationship_extractors = {}
    
    def extract_entities(self, text: str) -> List[Dict[str, Any]]:
        """
        Extract entities from text
        
        Parameters
        ----------
        text : str
            Input text
            
        Returns
        -------
        entities : list of dict
            Extracted entities with type and properties
        """
        # Simple entity extraction (can be enhanced with NER models)
        entities = []
        
        # Common ML entities
        ml_keywords = {
            'algorithm': ['random forest', 'svm', 'neural network', 'logistic regression'],
            'task': ['classification', 'regression', 'clustering'],
            'metric': ['accuracy', 'precision', 'recall', 'f1', 'r2'],
            'tool': ['scikit-learn', 'pytorch', 'tensorflow', 'pandas']
        }
        
        text_lower = text.lower()
        for entity_type, keywords in ml_keywords.items():
            for keyword in keywords:
                if keyword in text_lower:
                    entities.append({
                        'text': keyword,
                        'type': entity_type,
                        'start': text_lower.find(keyword),
                        'end': text_lower.find(keyword) + len(keyword)
                    })
        
        return entities
    
    def extract_relationships(self, text: str, entities: List[Dict]) -> List[Dict[str, Any]]:
        """
        Extract relationships between entities
        
        Parameters
        ----------
        text : str
            Input text
        entities : list of dict
            Extracted entities
            
        Returns
        -------
        relationships : list of dict
            Extracted relationships
        """
        relationships = []
        
        # Simple relationship extraction
        relationship_patterns = [
            ('uses', ['uses', 'employs', 'applies']),
            ('is_a', ['is a', 'is an', 'type of']),
            ('part_of', ['part of', 'component of']),
            ('related_to', ['related to', 'connected to', 'associated with'])
        ]
        
        text_lower = text.lower()
        for rel_type, patterns in relationship_patterns:
            for pattern in patterns:
                if pattern in text_lower:
                    # Find entities near the pattern
                    pattern_pos = text_lower.find(pattern)
                    for i, entity1 in enumerate(entities):
                        for entity2 in entities[i+1:]:
                            if abs(entity1['start'] - pattern_pos) < 50 or abs(entity2['start'] - pattern_pos) < 50:
                                relationships.append({
                                    'source': entity1['text'],
                                    'target': entity2['text'],
                                    'relationship': rel_type,
                                    'confidence': 0.7
                                })
        
        return relationships
    
    def build_from_text(self, text: str, doc_id: str):
        """
        Build knowledge graph from text
        
        Parameters
        ----------
        text : str
            Input text
        doc_id : str
            Document identifier
        """
        # Extract entities
        entities = self.extract_entities(text)
        
        # Add entities as nodes
        for i, entity in enumerate(entities):
            node_id = f"{doc_id}_entity_{i}"
            self.graph.add_node(
                node_id=node_id,
                node_type=entity['type'],
                properties={
                    'name': entity['text'],
                    'source': doc_id
                }
            )
        
        # Extract relationships
        relationships = self.extract_relationships(text, entities)
        
        # Add relationships as edges
        for rel in relationships:
            source_id = self._find_node_id(rel['source'])
            target_id = self._find_node_id(rel['target'])
            
            if source_id and target_id:
                self.graph.add_edge(
                    source_id=source_id,
                    target_id=target_id,
                    relationship=rel['relationship'],
                    properties={'confidence': rel['confidence']}
                )
    
    def _find_node_id(self, node_name: str) -> Optional[str]:
        """Find node ID by name"""
        return self.graph.node_index.get(node_name)
    
    def query_graph(self, query: str) -> Dict:
        """
        Query knowledge graph with natural language
        
        Parameters
        ----------
        query : str
            Natural language query
            
        Returns
        -------
        results : dict
            Query results
        """
        query_lower = query.lower()
        
        # Simple query parsing
        if 'find' in query_lower or 'get' in query_lower:
            # Extract entity name
            words = query.split()
            for i, word in enumerate(words):
                if word.lower() in ['find', 'get'] and i + 1 < len(words):
                    entity_name = words[i + 1]
                    results = self.graph.query('find_node', name=entity_name)
                    if results:
                        return {
                            'query': query,
                            'results': results,
                            'neighbors': self.graph.get_neighbors(results[0]['node_id'])
                        }
        
        # Default: return graph statistics
        return {
            'query': query,
            'statistics': self.graph.get_statistics()
        }
    
    def get_graph(self) -> KnowledgeGraph:
        """Get the knowledge graph"""
        return self.graph
