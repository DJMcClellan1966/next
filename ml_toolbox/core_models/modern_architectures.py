"""
Modern Neural Network Architectures

Implements:
- Transformer (Attention Mechanism)
- BERT Architecture
- GPT Architecture
"""
import numpy as np
from typing import Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)

# Try to import torch for better performance
try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False
    logger.warning("PyTorch not available. Using NumPy implementation.")


class MultiHeadAttention:
    """
    Multi-Head Attention Mechanism
    
    Core component of Transformer architecture
    """
    
    def __init__(self, d_model: int, n_heads: int = 8):
        """
        Initialize multi-head attention
        
        Parameters
        ----------
        d_model : int
            Model dimension
        n_heads : int
            Number of attention heads
        """
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        # Weight matrices
        self.W_q = np.random.randn(d_model, d_model) * 0.02
        self.W_k = np.random.randn(d_model, d_model) * 0.02
        self.W_v = np.random.randn(d_model, d_model) * 0.02
        self.W_o = np.random.randn(d_model, d_model) * 0.02
    
    def forward(self, query: np.ndarray, key: np.ndarray, value: np.ndarray,
               mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Forward pass
        
        Parameters
        ----------
        query : array, shape (batch, seq_len, d_model)
            Query vectors
        key : array
            Key vectors
        value : array
            Value vectors
        mask : array, optional
            Attention mask
            
        Returns
        -------
        output : array
            Attention output
        attention_weights : array
            Attention weights
        """
        batch_size, seq_len, _ = query.shape
        
        # Linear projections
        Q = query @ self.W_q  # (batch, seq_len, d_model)
        K = key @ self.W_k
        V = value @ self.W_v
        
        # Reshape for multi-head
        Q = Q.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        K = K.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        V = V.reshape(batch_size, seq_len, self.n_heads, self.d_k).transpose(0, 2, 1, 3)
        
        # Scaled dot-product attention
        scores = Q @ K.transpose(0, 1, 3, 2) / np.sqrt(self.d_k)
        
        # Apply mask
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attention_weights = self._softmax(scores, axis=-1)
        
        # Apply attention to values
        context = attention_weights @ V
        
        # Concatenate heads
        context = context.transpose(0, 2, 1, 3).reshape(batch_size, seq_len, self.d_model)
        
        # Output projection
        output = context @ self.W_o
        
        return output, attention_weights
    
    def _softmax(self, x: np.ndarray, axis: int = -1) -> np.ndarray:
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=axis, keepdims=True))
        return exp_x / np.sum(exp_x, axis=axis, keepdims=True)


class TransformerBlock:
    """
    Transformer Block
    
    Contains:
    - Multi-head attention
    - Feed-forward network
    - Layer normalization
    - Residual connections
    """
    
    def __init__(self, d_model: int, n_heads: int = 8, d_ff: int = 2048,
                 dropout: float = 0.1):
        """
        Initialize transformer block
        
        Parameters
        ----------
        d_model : int
            Model dimension
        n_heads : int
            Number of attention heads
        d_ff : int
            Feed-forward dimension
        dropout : float
            Dropout rate
        """
        self.attention = MultiHeadAttention(d_model, n_heads)
        self.norm1 = np.ones(d_model)  # Layer norm parameters (simplified)
        self.norm2 = np.ones(d_model)
        
        # Feed-forward network
        self.ff_weights1 = np.random.randn(d_model, d_ff) * 0.02
        self.ff_bias1 = np.zeros(d_ff)
        self.ff_weights2 = np.random.randn(d_ff, d_model) * 0.02
        self.ff_bias2 = np.zeros(d_model)
        
        self.dropout_rate = dropout
    
    def forward(self, x: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass
        
        Parameters
        ----------
        x : array
            Input
        mask : array, optional
            Attention mask
            
        Returns
        -------
        output : array
            Transformer block output
        """
        # Self-attention with residual
        attn_output, _ = self.attention.forward(x, x, x, mask)
        x = x + attn_output  # Residual connection
        x = self._layer_norm(x)  # Layer norm
        
        # Feed-forward with residual
        ff_output = self._feed_forward(x)
        x = x + ff_output  # Residual connection
        x = self._layer_norm(x)  # Layer norm
        
        return x
    
    def _layer_norm(self, x: np.ndarray) -> np.ndarray:
        """Layer normalization"""
        mean = np.mean(x, axis=-1, keepdims=True)
        var = np.var(x, axis=-1, keepdims=True)
        return (x - mean) / np.sqrt(var + 1e-5)
    
    def _feed_forward(self, x: np.ndarray) -> np.ndarray:
        """Feed-forward network"""
        # First layer
        h = x @ self.ff_weights1 + self.ff_bias1
        h = np.maximum(0, h)  # ReLU
        # Second layer
        return h @ self.ff_weights2 + self.ff_bias2


class Transformer:
    """
    Transformer Model
    
    Full transformer architecture with encoder and decoder
    """
    
    def __init__(self, vocab_size: int, d_model: int = 512, n_heads: int = 8,
                 n_layers: int = 6, d_ff: int = 2048, max_seq_length: int = 512,
                 dropout: float = 0.1):
        """
        Initialize transformer
        
        Parameters
        ----------
        vocab_size : int
            Vocabulary size
        d_model : int
            Model dimension
        n_heads : int
            Number of attention heads
        n_layers : int
            Number of transformer layers
        d_ff : int
            Feed-forward dimension
        max_seq_length : int
            Maximum sequence length
        dropout : float
            Dropout rate
        """
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.max_seq_length = max_seq_length
        
        # Embeddings
        self.token_embedding = np.random.randn(vocab_size, d_model) * 0.02
        self.position_embedding = np.random.randn(max_seq_length, d_model) * 0.02
        
        # Transformer blocks
        self.encoder_layers = [
            TransformerBlock(d_model, n_heads, d_ff, dropout)
            for _ in range(n_layers)
        ]
        
        # Output projection
        self.output_projection = np.random.randn(d_model, vocab_size) * 0.02
    
    def forward(self, input_ids: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass
        
        Parameters
        ----------
        input_ids : array, shape (batch, seq_len)
            Input token IDs
        mask : array, optional
            Attention mask
            
        Returns
        -------
        logits : array
            Output logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Token embeddings
        x = self.token_embedding[input_ids]  # (batch, seq_len, d_model)
        
        # Position embeddings
        positions = np.arange(seq_len)
        x = x + self.position_embedding[positions]
        
        # Apply transformer blocks
        for layer in self.encoder_layers:
            x = layer.forward(x, mask)
        
        # Output projection
        logits = x @ self.output_projection
        
        return logits


class BERT(Transformer):
    """
    BERT (Bidirectional Encoder Representations from Transformers)
    
    Masked language model architecture
    """
    
    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12,
                 n_layers: int = 12, d_ff: int = 3072, max_seq_length: int = 512,
                 dropout: float = 0.1):
        """
        Initialize BERT
        
        Parameters match BERT-base architecture
        """
        super().__init__(vocab_size, d_model, n_heads, n_layers, d_ff, 
                        max_seq_length, dropout)
        
        # BERT-specific: CLS token embedding
        self.cls_token = np.random.randn(1, d_model) * 0.02
        
        # Pooler for CLS token
        self.pooler_weights = np.random.randn(d_model, d_model) * 0.02
        self.pooler_bias = np.zeros(d_model)
    
    def forward(self, input_ids: np.ndarray, mask: Optional[np.ndarray] = None,
               return_cls: bool = False) -> np.ndarray:
        """
        Forward pass with BERT-specific features
        
        Parameters
        ----------
        input_ids : array
            Input token IDs
        mask : array, optional
            Attention mask
        return_cls : bool
            Whether to return CLS token representation
            
        Returns
        -------
        output : array
            BERT output
        """
        batch_size, seq_len = input_ids.shape
        
        # Add CLS token
        cls_emb = np.tile(self.cls_token, (batch_size, 1, 1))
        x = self.token_embedding[input_ids]
        x = np.concatenate([cls_emb, x], axis=1)
        
        # Position embeddings
        positions = np.arange(seq_len + 1)
        x = x + self.position_embedding[positions[:seq_len+1]]
        
        # Apply transformer blocks
        for layer in self.encoder_layers:
            x = layer.forward(x, mask)
        
        if return_cls:
            # Return CLS token representation
            cls_output = x[:, 0, :]  # First token is CLS
            pooled = np.tanh(cls_output @ self.pooler_weights + self.pooler_bias)
            return pooled
        
        # Output projection
        logits = x[:, 1:, :] @ self.output_projection  # Skip CLS token for output
        return logits


class GPT(Transformer):
    """
    GPT (Generative Pre-trained Transformer)
    
    Autoregressive language model architecture
    """
    
    def __init__(self, vocab_size: int, d_model: int = 768, n_heads: int = 12,
                 n_layers: int = 12, d_ff: int = 3072, max_seq_length: int = 1024,
                 dropout: float = 0.1):
        """
        Initialize GPT
        
        Parameters match GPT-2 architecture
        """
        super().__init__(vocab_size, d_model, n_heads, n_layers, d_ff,
                        max_seq_length, dropout)
    
    def forward(self, input_ids: np.ndarray, mask: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Forward pass with causal masking (autoregressive)
        
        Parameters
        ----------
        input_ids : array
            Input token IDs
        mask : array, optional
            Attention mask (will be overridden with causal mask)
            
        Returns
        -------
        logits : array
            Output logits
        """
        batch_size, seq_len = input_ids.shape
        
        # Create causal mask (lower triangular)
        causal_mask = np.triu(np.ones((seq_len, seq_len)), k=1)
        causal_mask = (causal_mask == 0).astype(float)
        
        # Token embeddings
        x = self.token_embedding[input_ids]
        
        # Position embeddings
        positions = np.arange(seq_len)
        x = x + self.position_embedding[positions]
        
        # Apply transformer blocks with causal mask
        for layer in self.encoder_layers:
            x = layer.forward(x, causal_mask)
        
        # Output projection
        logits = x @ self.output_projection
        
        return logits
    
    def generate(self, input_ids: np.ndarray, max_length: int = 50,
                temperature: float = 1.0) -> np.ndarray:
        """
        Generate text autoregressively
        
        Parameters
        ----------
        input_ids : array
            Initial token IDs
        max_length : int
            Maximum generation length
        temperature : float
            Sampling temperature
            
        Returns
        -------
        generated_ids : array
            Generated token IDs
        """
        generated = input_ids.copy()
        
        for _ in range(max_length - input_ids.shape[1]):
            # Forward pass
            logits = self.forward(generated)
            
            # Get next token logits
            next_token_logits = logits[:, -1, :] / temperature
            
            # Softmax
            probs = self._softmax(next_token_logits)
            
            # Sample
            next_token = np.random.choice(self.vocab_size, p=probs[0])
            generated = np.concatenate([generated, np.array([[next_token]])], axis=1)
        
        return generated
    
    def _softmax(self, x: np.ndarray) -> np.ndarray:
        """Softmax function"""
        exp_x = np.exp(x - np.max(x, axis=-1, keepdims=True))
        return exp_x / np.sum(exp_x, axis=-1, keepdims=True)
