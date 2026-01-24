"""
Advanced Deep Learning (Deep Learning - Goodfellow et al.)

Implements:
- Regularization Techniques
- Advanced Optimization
- Generative Models
- Attention Mechanisms
- Transfer Learning
"""
import numpy as np
from typing import Dict, List, Tuple, Optional, Any
import logging

logger = logging.getLogger(__name__)


class RegularizationTechniques:
    """
    Regularization Techniques for Deep Learning
    """
    
    @staticmethod
    def l1_regularization(weights: np.ndarray, lambda_reg: float = 0.01) -> float:
        """L1 regularization (Lasso)"""
        return lambda_reg * np.sum(np.abs(weights))
    
    @staticmethod
    def l2_regularization(weights: np.ndarray, lambda_reg: float = 0.01) -> float:
        """L2 regularization (Ridge)"""
        return lambda_reg * np.sum(weights ** 2)
    
    @staticmethod
    def elastic_net(weights: np.ndarray, lambda_l1: float = 0.01,
                   lambda_l2: float = 0.01) -> float:
        """Elastic net regularization"""
        return (RegularizationTechniques.l1_regularization(weights, lambda_l1) +
                RegularizationTechniques.l2_regularization(weights, lambda_l2))
    
    @staticmethod
    def early_stopping(validation_losses: List[float], patience: int = 5) -> bool:
        """
        Early stopping
        
        Parameters
        ----------
        validation_losses : list
            Validation losses over epochs
        patience : int
            Patience (epochs without improvement)
            
        Returns
        -------
        should_stop : bool
            Whether to stop training
        """
        if len(validation_losses) < patience + 1:
            return False
        
        best_loss = min(validation_losses[:-patience])
        recent_losses = validation_losses[-patience:]
        
        return all(loss >= best_loss for loss in recent_losses)
    
    @staticmethod
    def data_augmentation(images: np.ndarray, augmentation_type: str = 'rotation') -> np.ndarray:
        """Data augmentation"""
        augmented = images.copy()
        
        if augmentation_type == 'rotation':
            # Simple rotation (90 degrees)
            for i in range(len(images)):
                if np.random.random() < 0.5:
                    augmented[i] = np.rot90(images[i], k=1)
        elif augmentation_type == 'flip':
            # Horizontal flip
            for i in range(len(images)):
                if np.random.random() < 0.5:
                    augmented[i] = np.fliplr(images[i])
        
        return augmented


class AdvancedOptimization:
    """
    Advanced Optimization for Deep Learning
    """
    
    @staticmethod
    def rmsprop(gradients: np.ndarray, cache: np.ndarray, learning_rate: float = 0.001,
               decay_rate: float = 0.9, eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        RMSprop optimizer
        
        Parameters
        ----------
        gradients : array
            Gradients
        cache : array
            Running average of squared gradients
        learning_rate : float
            Learning rate
        decay_rate : float
            Decay rate
        eps : float
            Small value
            
        Returns
        -------
        update : array
            Parameter update
        cache : array
            Updated cache
        """
        cache = decay_rate * cache + (1 - decay_rate) * gradients ** 2
        update = learning_rate * gradients / (np.sqrt(cache) + eps)
        return update, cache
    
    @staticmethod
    def adagrad(gradients: np.ndarray, cache: np.ndarray, learning_rate: float = 0.01,
               eps: float = 1e-8) -> Tuple[np.ndarray, np.ndarray]:
        """
        Adagrad optimizer
        
        Parameters
        ----------
        gradients : array
            Gradients
        cache : array
            Accumulated squared gradients
        learning_rate : float
            Learning rate
        eps : float
            Small value
            
        Returns
        -------
        update : array
            Parameter update
        cache : array
            Updated cache
        """
        cache += gradients ** 2
        update = learning_rate * gradients / (np.sqrt(cache) + eps)
        return update, cache
    
    @staticmethod
    def learning_rate_schedule(initial_lr: float, epoch: int,
                            schedule_type: str = 'exponential',
                            decay_rate: float = 0.95) -> float:
        """
        Learning rate scheduling
        
        Parameters
        ----------
        initial_lr : float
            Initial learning rate
        epoch : int
            Current epoch
        schedule_type : str
            Schedule type ('exponential', 'step', 'cosine')
        decay_rate : float
            Decay rate
            
        Returns
        -------
        lr : float
            Learning rate for epoch
        """
        if schedule_type == 'exponential':
            return initial_lr * (decay_rate ** epoch)
        elif schedule_type == 'step':
            return initial_lr * (decay_rate ** (epoch // 10))
        elif schedule_type == 'cosine':
            return initial_lr * 0.5 * (1 + np.cos(np.pi * epoch / 100))
        else:
            return initial_lr


class GenerativeModels:
    """
    Generative Models
    
    GANs, VAEs, etc.
    """
    
    @staticmethod
    def gan_loss(real_pred: np.ndarray, fake_pred: np.ndarray) -> Tuple[float, float]:
        """
        GAN loss (adversarial loss)
        
        Parameters
        ----------
        real_pred : array
            Discriminator predictions on real data
        fake_pred : array
            Discriminator predictions on fake data
            
        Returns
        -------
        d_loss : float
            Discriminator loss
        g_loss : float
            Generator loss
        """
        # Discriminator: maximize log(D(real)) + log(1 - D(fake))
        d_loss = -(np.mean(np.log(real_pred + 1e-10)) +
                  np.mean(np.log(1 - fake_pred + 1e-10)))
        
        # Generator: maximize log(D(fake))
        g_loss = -np.mean(np.log(fake_pred + 1e-10))
        
        return d_loss, g_loss
    
    @staticmethod
    def vae_loss(reconstructed: np.ndarray, original: np.ndarray,
                mu: np.ndarray, logvar: np.ndarray, beta: float = 1.0) -> float:
        """
        VAE loss (reconstruction + KL divergence)
        
        Parameters
        ----------
        reconstructed : array
            Reconstructed data
        original : array
            Original data
        mu : array
            Mean of latent distribution
        logvar : array
            Log variance of latent distribution
        beta : float
            KL weight
            
        Returns
        -------
        loss : float
            VAE loss
        """
        # Reconstruction loss (MSE)
        recon_loss = np.mean((reconstructed - original) ** 2)
        
        # KL divergence: -0.5 * sum(1 + logvar - mu^2 - exp(logvar))
        kl_loss = -0.5 * np.mean(1 + logvar - mu ** 2 - np.exp(logvar))
        
        return recon_loss + beta * kl_loss


class AttentionMechanisms:
    """
    Attention Mechanisms
    
    Self-attention, multi-head attention
    """
    
    @staticmethod
    def scaled_dot_product_attention(Q: np.ndarray, K: np.ndarray, V: np.ndarray,
                                   mask: Optional[np.ndarray] = None) -> Tuple[np.ndarray, np.ndarray]:
        """
        Scaled dot-product attention
        
        Parameters
        ----------
        Q : array
            Query matrix
        K : array
            Key matrix
        V : array
            Value matrix
        mask : array, optional
            Attention mask
            
        Returns
        -------
        output : array
            Attention output
        attention_weights : array
            Attention weights
        """
        d_k = Q.shape[-1]
        
        # Compute attention scores
        scores = Q @ K.T / np.sqrt(d_k)
        
        # Apply mask
        if mask is not None:
            scores = np.where(mask == 0, -1e9, scores)
        
        # Softmax
        attention_weights = np.exp(scores - np.max(scores, axis=-1, keepdims=True))
        attention_weights = attention_weights / np.sum(attention_weights, axis=-1, keepdims=True)
        
        # Apply to values
        output = attention_weights @ V
        
        return output, attention_weights
    
    @staticmethod
    def self_attention(X: np.ndarray, W_q: np.ndarray, W_k: np.ndarray,
                      W_v: np.ndarray) -> np.ndarray:
        """
        Self-attention
        
        Parameters
        ----------
        X : array
            Input sequence
        W_q : array
            Query weights
        W_k : array
            Key weights
        W_v : array
            Value weights
            
        Returns
        -------
        output : array
            Self-attention output
        """
        Q = X @ W_q
        K = X @ W_k
        V = X @ W_v
        
        output, _ = AttentionMechanisms.scaled_dot_product_attention(Q, K, V)
        return output


class TransferLearning:
    """
    Transfer Learning
    
    Fine-tuning pre-trained models
    """
    
    @staticmethod
    def freeze_layers(model: Any, n_layers_to_freeze: int):
        """
        Freeze layers for fine-tuning
        
        Parameters
        ----------
        model : any
            Model with layers
        n_layers_to_freeze : int
            Number of layers to freeze
        """
        # Simplified: would set requires_grad=False in PyTorch
        logger.info(f"[TransferLearning] Freezing {n_layers_to_freeze} layers")
    
    @staticmethod
    def fine_tune(model: Any, X_new: np.ndarray, y_new: np.ndarray,
                 learning_rate: float = 0.0001, epochs: int = 10):
        """
        Fine-tune model on new data
        
        Parameters
        ----------
        model : any
            Pre-trained model
        X_new : array
            New training data
        y_new : array
            New target values
        learning_rate : float
            Learning rate (usually smaller)
        epochs : int
            Number of epochs
        """
        # Simplified fine-tuning
        logger.info(f"[TransferLearning] Fine-tuning for {epochs} epochs")
        # In practice, would train with lower learning rate
