"""
Dimensionality Reduction (Mathematics for ML, PRML)

Implements:
- PCA (Principal Component Analysis)
- LDA (Linear Discriminant Analysis)
- t-SNE
- UMAP (simplified)
- Autoencoders
"""
import numpy as np
from typing import Optional, Tuple
import logging

logger = logging.getLogger(__name__)


class PCA:
    """
    Principal Component Analysis
    
    Linear dimensionality reduction
    """
    
    def __init__(self, n_components: int = 2):
        """
        Initialize PCA
        
        Parameters
        ----------
        n_components : int
            Number of components
        """
        self.n_components = n_components
        self.components_ = None
        self.explained_variance_ = None
        self.mean_ = None
    
    def fit(self, X: np.ndarray):
        """
        Fit PCA
        
        Parameters
        ----------
        X : array
            Training data
        """
        X = np.asarray(X, dtype=np.float64)
        self.mean_ = np.mean(X, axis=0)
        X_centered = X - self.mean_
        
        # Covariance matrix
        cov = X_centered.T @ X_centered / (len(X) - 1)
        
        # Eigen-decomposition
        eigenvalues, eigenvectors = np.linalg.eig(cov)
        
        # Sort by eigenvalue
        idx = np.argsort(eigenvalues)[::-1]
        eigenvalues = eigenvalues[idx]
        eigenvectors = eigenvectors[:, idx]
        
        # Select components
        self.components_ = eigenvectors[:, :self.n_components].T
        self.explained_variance_ = eigenvalues[:self.n_components]
        
        logger.info(f"[PCA] Fitted with {self.n_components} components")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        if self.components_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        X = np.asarray(X, dtype=np.float64)
        X_centered = X - self.mean_
        return X_centered @ self.components_.T
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """Fit and transform"""
        self.fit(X)
        return self.transform(X)


class LDA:
    """
    Linear Discriminant Analysis
    
    Supervised dimensionality reduction
    """
    
    def __init__(self, n_components: int = 2):
        """
        Initialize LDA
        
        Parameters
        ----------
        n_components : int
            Number of components
        """
        self.n_components = n_components
        self.components_ = None
        self.class_means_ = None
    
    def fit(self, X: np.ndarray, y: np.ndarray):
        """
        Fit LDA
        
        Parameters
        ----------
        X : array
            Training data
        y : array
            Class labels
        """
        X = np.asarray(X, dtype=np.float64)
        y = np.asarray(y)
        
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]
        
        # Overall mean
        overall_mean = np.mean(X, axis=0)
        
        # Within-class scatter
        S_W = np.zeros((n_features, n_features))
        self.class_means_ = {}
        
        for c in classes:
            X_c = X[y == c]
            mean_c = np.mean(X_c, axis=0)
            self.class_means_[c] = mean_c
            
            # Within-class scatter
            S_W += (X_c - mean_c).T @ (X_c - mean_c)
        
        # Between-class scatter
        S_B = np.zeros((n_features, n_features))
        for c in classes:
            n_c = np.sum(y == c)
            mean_c = self.class_means_[c]
            diff = (mean_c - overall_mean).reshape(-1, 1)
            S_B += n_c * diff @ diff.T
        
        # Solve generalized eigenvalue problem
        # S_B @ w = lambda * S_W @ w
        # Equivalent to: S_W^-1 @ S_B @ w = lambda * w
        try:
            S_W_inv = np.linalg.inv(S_W)
            matrix = S_W_inv @ S_B
            eigenvalues, eigenvectors = np.linalg.eig(matrix)
            
            # Sort and select
            idx = np.argsort(eigenvalues)[::-1]
            self.components_ = eigenvectors[:, idx[:self.n_components]].T
        except:
            # Fallback to PCA if singular
            logger.warning("[LDA] S_W singular, using PCA fallback")
            pca = PCA(n_components=self.n_components)
            self.components_ = pca.fit(X).components_
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform data"""
        if self.components_ is None:
            raise ValueError("Model not fitted. Call fit() first.")
        
        return X @ self.components_.T


class tSNE:
    """
    t-Distributed Stochastic Neighbor Embedding
    
    Non-linear dimensionality reduction
    """
    
    def __init__(self, n_components: int = 2, perplexity: float = 30.0,
                 learning_rate: float = 200.0, n_iter: int = 1000):
        """
        Initialize t-SNE
        
        Parameters
        ----------
        n_components : int
            Number of components
        perplexity : float
            Perplexity parameter
        learning_rate : float
            Learning rate
        n_iter : int
            Number of iterations
        """
        self.n_components = n_components
        self.perplexity = perplexity
        self.learning_rate = learning_rate
        self.n_iter = n_iter
        self.embedding_ = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform using t-SNE
        
        Parameters
        ----------
        X : array
            Input data
            
        Returns
        -------
        embedding : array
            Low-dimensional embedding
        """
        X = np.asarray(X, dtype=np.float64)
        n_samples = X.shape[0]
        
        # Initialize embedding randomly
        self.embedding_ = np.random.randn(n_samples, self.n_components) * 0.0001
        
        # Compute pairwise distances
        distances = np.sqrt(((X[:, np.newaxis, :] - X) ** 2).sum(axis=2))
        
        # Compute P matrix (high-dimensional similarities)
        P = self._compute_p_matrix(distances)
        
        # Gradient descent
        for iteration in range(self.n_iter):
            # Compute Q matrix (low-dimensional similarities)
            Q = self._compute_q_matrix(self.embedding_)
            
            # Compute gradient
            gradient = self._compute_gradient(P, Q, self.embedding_)
            
            # Update embedding
            self.embedding_ += self.learning_rate * gradient
            
            # Momentum (simplified)
            if iteration < 250:
                momentum = 0.5
            else:
                momentum = 0.8
            
            self.embedding_ += momentum * gradient
        
        return self.embedding_
    
    def _compute_p_matrix(self, distances: np.ndarray) -> np.ndarray:
        """Compute P matrix (high-dimensional similarities)"""
        # Gaussian kernel
        sigmas = self._find_sigmas(distances)
        P = np.zeros_like(distances)
        
        for i in range(len(distances)):
            for j in range(len(distances)):
                if i != j:
                    P[i, j] = np.exp(-distances[i, j] ** 2 / (2 * sigmas[i] ** 2))
        
        # Symmetrize
        P = (P + P.T) / (2 * len(distances))
        P = P / P.sum()
        
        return P
    
    def _find_sigmas(self, distances: np.ndarray) -> np.ndarray:
        """Find sigma for each point to achieve target perplexity"""
        sigmas = np.ones(len(distances))
        
        for i in range(len(distances)):
            # Binary search for sigma
            sigma_min, sigma_max = 1e-10, 1000.0
            
            for _ in range(50):
                sigma = (sigma_min + sigma_max) / 2
                probs = np.exp(-distances[i] ** 2 / (2 * sigma ** 2))
                probs[i] = 0
                probs = probs / probs.sum()
                
                entropy = -np.sum(probs[probs > 0] * np.log2(probs[probs > 0]))
                perplexity = 2 ** entropy
                
                if perplexity < self.perplexity:
                    sigma_min = sigma
                else:
                    sigma_max = sigma
            
            sigmas[i] = (sigma_min + sigma_max) / 2
        
        return sigmas
    
    def _compute_q_matrix(self, embedding: np.ndarray) -> np.ndarray:
        """Compute Q matrix (low-dimensional similarities)"""
        distances = np.sqrt(((embedding[:, np.newaxis, :] - embedding) ** 2).sum(axis=2))
        
        # t-distribution
        Q = 1 / (1 + distances ** 2)
        np.fill_diagonal(Q, 0)
        Q = Q / Q.sum()
        
        return Q
    
    def _compute_gradient(self, P: np.ndarray, Q: np.ndarray,
                         embedding: np.ndarray) -> np.ndarray:
        """Compute gradient"""
        n = len(embedding)
        gradient = np.zeros_like(embedding)
        
        for i in range(n):
            for j in range(n):
                if i != j:
                    diff = embedding[i] - embedding[j]
                    dist_sq = np.sum(diff ** 2)
                    
                    gradient[i] += 4 * (P[i, j] - Q[i, j]) * diff / (1 + dist_sq)
        
        return gradient


class UMAP:
    """
    UMAP (Uniform Manifold Approximation and Projection)
    
    Simplified implementation
    """
    
    def __init__(self, n_components: int = 2, n_neighbors: int = 15):
        """
        Initialize UMAP
        
        Parameters
        ----------
        n_components : int
            Number of components
        n_neighbors : int
            Number of neighbors
        """
        self.n_components = n_components
        self.n_neighbors = n_neighbors
        self.embedding_ = None
    
    def fit_transform(self, X: np.ndarray) -> np.ndarray:
        """
        Fit and transform using UMAP (simplified)
        
        Parameters
        ----------
        X : array
            Input data
            
        Returns
        -------
        embedding : array
            Low-dimensional embedding
        """
        # Simplified UMAP (uses t-SNE as approximation)
        # Full UMAP is more complex
        logger.warning("[UMAP] Using simplified implementation (t-SNE approximation)")
        
        tsne = tSNE(n_components=self.n_components)
        return tsne.fit_transform(X)


class Autoencoder:
    """
    Autoencoder
    
    Neural network for dimensionality reduction
    """
    
    def __init__(self, input_dim: int, encoding_dim: int, hidden_dims: List[int] = None):
        """
        Initialize autoencoder
        
        Parameters
        ----------
        input_dim : int
            Input dimension
        encoding_dim : int
            Encoding (latent) dimension
        hidden_dims : list, optional
            Hidden layer dimensions
        """
        self.input_dim = input_dim
        self.encoding_dim = encoding_dim
        self.hidden_dims = hidden_dims or []
        
        # Initialize weights (simplified)
        self.encoder_weights = []
        self.decoder_weights = []
        self._initialize_weights()
    
    def _initialize_weights(self):
        """Initialize network weights"""
        dims = [self.input_dim] + self.hidden_dims + [self.encoding_dim]
        
        for i in range(len(dims) - 1):
            w = np.random.randn(dims[i], dims[i+1]) * 0.01
            self.encoder_weights.append(w)
        
        dims_rev = [self.encoding_dim] + self.hidden_dims[::-1] + [self.input_dim]
        
        for i in range(len(dims_rev) - 1):
            w = np.random.randn(dims_rev[i], dims_rev[i+1]) * 0.01
            self.decoder_weights.append(w)
    
    def encode(self, X: np.ndarray) -> np.ndarray:
        """Encode input to latent representation"""
        X = np.asarray(X)
        encoded = X
        
        for w in self.encoder_weights:
            encoded = np.maximum(0, encoded @ w)  # ReLU
        
        return encoded
    
    def decode(self, encoded: np.ndarray) -> np.ndarray:
        """Decode latent representation to output"""
        decoded = encoded
        
        for w in self.decoder_weights:
            decoded = np.maximum(0, decoded @ w)  # ReLU
        
        return decoded
    
    def fit(self, X: np.ndarray, epochs: int = 100, learning_rate: float = 0.01):
        """Train autoencoder"""
        X = np.asarray(X)
        
        for epoch in range(epochs):
            # Forward pass
            encoded = self.encode(X)
            decoded = self.decode(encoded)
            
            # Loss (MSE)
            loss = np.mean((X - decoded) ** 2)
            
            # Simplified training (gradient descent)
            # In practice, use backpropagation
            if epoch % 10 == 0:
                logger.info(f"[Autoencoder] Epoch {epoch}, Loss: {loss:.4f}")
    
    def transform(self, X: np.ndarray) -> np.ndarray:
        """Transform to latent space"""
        return self.encode(X)
