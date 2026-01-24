"""
Clustering Algorithms

Implements:
- K-Means
- DBSCAN
- Hierarchical Clustering
"""
import numpy as np
from typing import List, Dict, Tuple, Optional
from collections import defaultdict
import logging

logger = logging.getLogger(__name__)


class KMeans:
    """
    K-Means Clustering
    
    Partition data into k clusters
    """
    
    def __init__(self, n_clusters: int = 3, max_iter: int = 300, 
                 init: str = 'random', random_state: Optional[int] = None):
        """
        Initialize K-Means
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        max_iter : int
            Maximum iterations
        init : str
            Initialization method ('random' or 'kmeans++')
        random_state : int, optional
            Random seed
        """
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.init = init
        self.random_state = random_state
        self.centroids_ = None
        self.labels_ = None
    
    def _initialize_centroids(self, X: np.ndarray) -> np.ndarray:
        """Initialize centroids"""
        if self.random_state is not None:
            np.random.seed(self.random_state)
        
        if self.init == 'kmeans++':
            # K-means++ initialization
            n_samples, n_features = X.shape
            centroids = np.zeros((self.n_clusters, n_features))
            
            # First centroid: random point
            centroids[0] = X[np.random.randint(n_samples)]
            
            # Subsequent centroids: farthest from existing centroids
            for i in range(1, self.n_clusters):
                distances = np.array([
                    min([np.linalg.norm(x - c) ** 2 for c in centroids[:i]])
                    for x in X
                ])
                probabilities = distances / distances.sum()
                centroids[i] = X[np.random.choice(n_samples, p=probabilities)]
            
            return centroids
        else:
            # Random initialization
            indices = np.random.choice(X.shape[0], self.n_clusters, replace=False)
            return X[indices]
    
    def fit(self, X: np.ndarray):
        """
        Fit K-Means clustering
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples, n_features = X.shape
        
        # Initialize centroids
        self.centroids_ = self._initialize_centroids(X)
        
        for iteration in range(self.max_iter):
            # Assign points to nearest centroid
            distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids_) ** 2).sum(axis=2))
            self.labels_ = np.argmin(distances, axis=1)
            
            # Update centroids
            new_centroids = np.zeros_like(self.centroids_)
            for k in range(self.n_clusters):
                cluster_points = X[self.labels_ == k]
                if len(cluster_points) > 0:
                    new_centroids[k] = cluster_points.mean(axis=0)
                else:
                    # Keep old centroid if cluster is empty
                    new_centroids[k] = self.centroids_[k]
            
            # Check convergence
            if np.allclose(self.centroids_, new_centroids):
                break
            
            self.centroids_ = new_centroids
        
        logger.info(f"[KMeans] Fitted {self.n_clusters} clusters in {iteration + 1} iterations")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict cluster labels
        
        Parameters
        ----------
        X : array
            Input data
            
        Returns
        -------
        labels : array
            Cluster labels
        """
        if self.centroids_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        distances = np.sqrt(((X[:, np.newaxis, :] - self.centroids_) ** 2).sum(axis=2))
        return np.argmin(distances, axis=1)


class DBSCAN:
    """
    DBSCAN Clustering
    
    Density-based clustering
    """
    
    def __init__(self, eps: float = 0.5, min_samples: int = 5):
        """
        Initialize DBSCAN
        
        Parameters
        ----------
        eps : float
            Maximum distance for neighbors
        min_samples : int
            Minimum samples in neighborhood
        """
        self.eps = eps
        self.min_samples = min_samples
        self.labels_ = None
    
    def _get_neighbors(self, X: np.ndarray, point_idx: int) -> List[int]:
        """Get neighbors within eps distance"""
        distances = np.linalg.norm(X - X[point_idx], axis=1)
        return np.where(distances <= self.eps)[0].tolist()
    
    def fit(self, X: np.ndarray):
        """
        Fit DBSCAN clustering
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        self.labels_ = np.full(n_samples, -1)  # -1 = noise
        cluster_id = 0
        
        for point_idx in range(n_samples):
            if self.labels_[point_idx] != -1:
                continue  # Already processed
            
            # Get neighbors
            neighbors = self._get_neighbors(X, point_idx)
            
            if len(neighbors) < self.min_samples:
                # Noise point
                self.labels_[point_idx] = -1
                continue
            
            # Start new cluster
            self.labels_[point_idx] = cluster_id
            
            # Expand cluster
            i = 0
            while i < len(neighbors):
                neighbor_idx = neighbors[i]
                
                if self.labels_[neighbor_idx] == -1:
                    # Change from noise to border point
                    self.labels_[neighbor_idx] = cluster_id
                elif self.labels_[neighbor_idx] == -1:
                    # Add to cluster
                    self.labels_[neighbor_idx] = cluster_id
                    
                    # Get neighbors of neighbor
                    neighbor_neighbors = self._get_neighbors(X, neighbor_idx)
                    if len(neighbor_neighbors) >= self.min_samples:
                        neighbors.extend(neighbor_neighbors)
                
                i += 1
            
            cluster_id += 1
        
        logger.info(f"[DBSCAN] Found {cluster_id} clusters with eps={self.eps}, min_samples={self.min_samples}")
    
    def fit_predict(self, X: np.ndarray) -> np.ndarray:
        """Fit and predict in one step"""
        self.fit(X)
        return self.labels_


class HierarchicalClustering:
    """
    Hierarchical Clustering
    
    Agglomerative clustering
    """
    
    def __init__(self, n_clusters: int = 2, linkage: str = 'ward'):
        """
        Initialize hierarchical clustering
        
        Parameters
        ----------
        n_clusters : int
            Number of clusters
        linkage : str
            Linkage criterion ('ward', 'complete', 'average', 'single')
        """
        self.n_clusters = n_clusters
        self.linkage = linkage
        self.labels_ = None
        self.children_ = None
    
    def _compute_distance(self, cluster1: np.ndarray, cluster2: np.ndarray) -> float:
        """Compute distance between clusters"""
        if self.linkage == 'ward':
            # Ward's method: minimize within-cluster variance
            return np.var(np.vstack([cluster1, cluster2])) * len(cluster1) * len(cluster2)
        elif self.linkage == 'complete':
            # Complete linkage: maximum distance
            distances = np.linalg.norm(cluster1[:, np.newaxis, :] - cluster2, axis=2)
            return np.max(distances)
        elif self.linkage == 'average':
            # Average linkage: average distance
            distances = np.linalg.norm(cluster1[:, np.newaxis, :] - cluster2, axis=2)
            return np.mean(distances)
        else:  # single
            # Single linkage: minimum distance
            distances = np.linalg.norm(cluster1[:, np.newaxis, :] - cluster2, axis=2)
            return np.min(distances)
    
    def fit(self, X: np.ndarray):
        """
        Fit hierarchical clustering
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Start with each point as its own cluster
        clusters = [[i] for i in range(n_samples)]
        cluster_data = {i: X[i:i+1] for i in range(n_samples)}
        
        # Merge clusters until we have n_clusters
        while len(clusters) > self.n_clusters:
            # Find closest pair of clusters
            min_dist = float('inf')
            merge_i, merge_j = 0, 1
            
            for i in range(len(clusters)):
                for j in range(i + 1, len(clusters)):
                    data_i = np.vstack([cluster_data[idx] for idx in clusters[i]])
                    data_j = np.vstack([cluster_data[idx] for idx in clusters[j]])
                    dist = self._compute_distance(data_i, data_j)
                    
                    if dist < min_dist:
                        min_dist = dist
                        merge_i, merge_j = i, j
            
            # Merge clusters
            clusters[merge_i].extend(clusters[merge_j])
            del clusters[merge_j]
        
        # Assign labels
        self.labels_ = np.zeros(n_samples, dtype=int)
        for cluster_id, cluster in enumerate(clusters):
            for point_idx in cluster:
                self.labels_[point_idx] = cluster_id
        
        logger.info(f"[HierarchicalClustering] Fitted {self.n_clusters} clusters using {self.linkage} linkage")
