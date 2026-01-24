"""
Neural Network Training Components

Implements:
- Neural Network
- Stochastic Gradient Descent (SGD)
- Dropout
- Batch Normalization
"""
import numpy as np
from typing import List, Optional, Tuple, Dict, Any
import logging

logger = logging.getLogger(__name__)


class Dropout:
    """
    Dropout Layer
    
    Randomly sets inputs to zero during training to prevent overfitting
    """
    
    def __init__(self, dropout_rate: float = 0.5):
        """
        Initialize dropout
        
        Parameters
        ----------
        dropout_rate : float
            Probability of dropping a neuron (0-1)
        """
        self.dropout_rate = dropout_rate
        self.mask = None
        self.training = True
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass
        
        Parameters
        ----------
        x : array
            Input data
        training : bool
            Whether in training mode
            
        Returns
        -------
        output : array
            Output with dropout applied
        """
        self.training = training
        
        if training and self.dropout_rate > 0:
            # Create dropout mask
            self.mask = np.random.binomial(1, 1 - self.dropout_rate, size=x.shape)
            # Scale by keep probability to maintain expected value
            return x * self.mask / (1 - self.dropout_rate)
        else:
            return x
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass
        
        Parameters
        ----------
        grad_output : array
            Gradient from next layer
            
        Returns
        -------
        grad_input : array
            Gradient to previous layer
        """
        if self.training and self.mask is not None:
            return grad_output * self.mask / (1 - self.dropout_rate)
        else:
            return grad_output


class BatchNormalization:
    """
    Batch Normalization Layer
    
    Normalizes inputs to have zero mean and unit variance
    """
    
    def __init__(self, num_features: int, momentum: float = 0.9, eps: float = 1e-5):
        """
        Initialize batch normalization
        
        Parameters
        ----------
        num_features : int
            Number of features
        momentum : float
            Momentum for running statistics
        eps : float
            Small value for numerical stability
        """
        self.num_features = num_features
        self.momentum = momentum
        self.eps = eps
        self.gamma = np.ones(num_features)  # Scale parameter
        self.beta = np.zeros(num_features)  # Shift parameter
        self.running_mean = np.zeros(num_features)
        self.running_var = np.ones(num_features)
        self.training = True
        self.x_normalized = None
        self.x_mean = None
        self.x_var = None
    
    def forward(self, x: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass
        
        Parameters
        ----------
        x : array, shape (batch_size, num_features)
            Input data
        training : bool
            Whether in training mode
            
        Returns
        -------
        output : array
            Normalized output
        """
        self.training = training
        
        if training:
            # Compute batch statistics
            self.x_mean = np.mean(x, axis=0)
            self.x_var = np.var(x, axis=0)
            
            # Update running statistics
            self.running_mean = (self.momentum * self.running_mean + 
                                (1 - self.momentum) * self.x_mean)
            self.running_var = (self.momentum * self.running_var + 
                              (1 - self.momentum) * self.x_var)
            
            # Normalize
            self.x_normalized = (x - self.x_mean) / np.sqrt(self.x_var + self.eps)
        else:
            # Use running statistics
            self.x_normalized = (x - self.running_mean) / np.sqrt(self.running_var + self.eps)
        
        # Scale and shift
        return self.gamma * self.x_normalized + self.beta
    
    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass
        
        Parameters
        ----------
        grad_output : array
            Gradient from next layer
            
        Returns
        -------
        grad_input : array
            Gradient to previous layer
        """
        if not self.training:
            return grad_output * self.gamma
        
        batch_size = grad_output.shape[0]
        
        # Gradients w.r.t. normalized input
        grad_normalized = grad_output * self.gamma
        
        # Gradients w.r.t. variance
        grad_var = np.sum(grad_normalized * (self.x_normalized), axis=0) * -0.5 * \
                  (self.x_var + self.eps) ** (-3/2)
        
        # Gradients w.r.t. mean
        grad_mean = np.sum(grad_normalized * -1 / np.sqrt(self.x_var + self.eps), axis=0) + \
                   grad_var * np.mean(-2 * (self.x_normalized * np.sqrt(self.x_var + self.eps)), axis=0)
        
        # Gradient w.r.t. input
        grad_input = grad_normalized / np.sqrt(self.x_var + self.eps) + \
                    grad_var * 2 * (self.x_normalized * np.sqrt(self.x_var + self.eps)) / batch_size + \
                    grad_mean / batch_size
        
        return grad_input


class SGD:
    """
    Stochastic Gradient Descent Optimizer
    """
    
    def __init__(self, learning_rate: float = 0.01, momentum: float = 0.0):
        """
        Initialize SGD
        
        Parameters
        ----------
        learning_rate : float
            Learning rate
        momentum : float
            Momentum factor
        """
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.velocity = {}
    
    def update(self, params: Dict[str, np.ndarray], 
              grads: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
        """
        Update parameters using SGD
        
        Parameters
        ----------
        params : dict
            Current parameters
        grads : dict
            Gradients
            
        Returns
        -------
        updated_params : dict
            Updated parameters
        """
        updated_params = {}
        
        for key in params:
            if key not in self.velocity:
                self.velocity[key] = np.zeros_like(params[key])
            
            # Update velocity with momentum
            self.velocity[key] = (self.momentum * self.velocity[key] - 
                                 self.learning_rate * grads[key])
            
            # Update parameters
            updated_params[key] = params[key] + self.velocity[key]
        
        return updated_params


class NeuralNetwork:
    """
    Multi-Layer Neural Network
    
    Supports:
    - Multiple hidden layers
    - Dropout
    - Batch Normalization
    - SGD optimization
    """
    
    def __init__(self, layers: List[int], activation: str = 'relu',
                 dropout_rate: float = 0.0, use_batch_norm: bool = False):
        """
        Initialize neural network
        
        Parameters
        ----------
        layers : list of int
            Number of neurons in each layer (including input and output)
        activation : str
            Activation function ('relu', 'sigmoid', 'tanh')
        dropout_rate : float
            Dropout rate
        use_batch_norm : bool
            Whether to use batch normalization
        """
        self.layers = layers
        self.activation = activation
        self.dropout_rate = dropout_rate
        self.use_batch_norm = use_batch_norm
        
        # Initialize weights and biases
        self.weights = []
        self.biases = []
        self.batch_norms = []
        self.dropouts = []
        
        for i in range(len(layers) - 1):
            # Xavier initialization
            w = np.random.randn(layers[i], layers[i+1]) * np.sqrt(2.0 / layers[i])
            b = np.zeros(layers[i+1])
            self.weights.append(w)
            self.biases.append(b)
            
            # Batch normalization
            if use_batch_norm and i < len(layers) - 2:  # Not on output layer
                self.batch_norms.append(BatchNormalization(layers[i+1]))
            else:
                self.batch_norms.append(None)
            
            # Dropout
            if dropout_rate > 0 and i < len(layers) - 2:  # Not on output layer
                self.dropouts.append(Dropout(dropout_rate))
            else:
                self.dropouts.append(None)
        
        self.optimizer = SGD(learning_rate=0.01)
        self.training = True
    
    def _activate(self, x: np.ndarray) -> np.ndarray:
        """Apply activation function"""
        if self.activation == 'relu':
            return np.maximum(0, x)
        elif self.activation == 'sigmoid':
            return 1 / (1 + np.exp(-np.clip(x, -500, 500)))
        elif self.activation == 'tanh':
            return np.tanh(x)
        else:
            return x
    
    def _activate_derivative(self, x: np.ndarray) -> np.ndarray:
        """Derivative of activation function"""
        if self.activation == 'relu':
            return (x > 0).astype(float)
        elif self.activation == 'sigmoid':
            s = self._activate(x)
            return s * (1 - s)
        elif self.activation == 'tanh':
            return 1 - np.tanh(x) ** 2
        else:
            return np.ones_like(x)
    
    def forward(self, X: np.ndarray, training: bool = True) -> np.ndarray:
        """
        Forward pass
        
        Parameters
        ----------
        X : array, shape (n_samples, n_features)
            Input data
        training : bool
            Whether in training mode
            
        Returns
        -------
        output : array
            Network output
        """
        self.training = training
        self.activations = [X]
        self.z_values = []
        
        current = X
        
        for i in range(len(self.weights)):
            # Linear transformation
            z = current @ self.weights[i] + self.biases[i]
            self.z_values.append(z)
            
            # Batch normalization
            if self.batch_norms[i] is not None:
                z = self.batch_norms[i].forward(z, training=training)
            
            # Activation
            if i < len(self.weights) - 1:  # Not output layer
                a = self._activate(z)
                
                # Dropout
                if self.dropouts[i] is not None:
                    a = self.dropouts[i].forward(a, training=training)
                
                current = a
                self.activations.append(current)
            else:
                # Output layer (no activation for regression, softmax for classification)
                current = z
                self.activations.append(current)
        
        return current
    
    def backward(self, X: np.ndarray, y: np.ndarray, output: np.ndarray) -> Dict[str, List[np.ndarray]]:
        """
        Backward pass (backpropagation)
        
        Parameters
        ----------
        X : array
            Input data
        y : array
            True labels
        output : array
            Network output
            
        Returns
        -------
        gradients : dict
            Gradients for weights and biases
        """
        m = X.shape[0]
        grads_w = []
        grads_b = []
        
        # Output layer gradient
        if output.shape[1] == 1:
            # Regression
            error = output - y.reshape(-1, 1)
            delta = error / m
        else:
            # Classification (softmax + cross-entropy)
            # Softmax
            exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
            softmax = exp_output / np.sum(exp_output, axis=1, keepdims=True)
            # One-hot encode y
            y_onehot = np.zeros_like(softmax)
            y_onehot[np.arange(m), y.astype(int)] = 1
            delta = (softmax - y_onehot) / m
        
        # Backpropagate
        for i in reversed(range(len(self.weights))):
            # Gradient w.r.t. weights and biases
            grad_w = self.activations[i].T @ delta
            grad_b = np.sum(delta, axis=0)
            
            grads_w.insert(0, grad_w)
            grads_b.insert(0, grad_b)
            
            if i > 0:
                # Gradient w.r.t. previous layer
                delta = delta @ self.weights[i].T
                
                # Batch normalization backward
                if self.batch_norms[i] is not None:
                    delta = self.batch_norms[i].backward(delta)
                
                # Activation derivative
                delta = delta * self._activate_derivative(self.z_values[i-1])
                
                # Dropout backward
                if self.dropouts[i-1] is not None:
                    delta = self.dropouts[i-1].backward(delta)
        
        return {'weights': grads_w, 'biases': grads_b}
    
    def fit(self, X: np.ndarray, y: np.ndarray, epochs: int = 100,
           batch_size: int = 32, learning_rate: float = 0.01):
        """
        Train neural network
        
        Parameters
        ----------
        X : array
            Training data
        y : array
            Training labels
        epochs : int
            Number of epochs
        batch_size : int
            Batch size
        learning_rate : float
            Learning rate
        """
        self.optimizer.learning_rate = learning_rate
        X = np.asarray(X)
        y = np.asarray(y)
        
        n_samples = X.shape[0]
        
        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0.0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i+batch_size]
                y_batch = y_shuffled[i:i+batch_size]
                
                # Forward pass
                output = self.forward(X_batch, training=True)
                
                # Compute loss
                if output.shape[1] == 1:
                    loss = np.mean((output.ravel() - y_batch) ** 2)
                else:
                    # Cross-entropy
                    exp_output = np.exp(output - np.max(output, axis=1, keepdims=True))
                    softmax = exp_output / np.sum(exp_output, axis=1, keepdims=True)
                    y_onehot = np.zeros_like(softmax)
                    y_onehot[np.arange(len(y_batch)), y_batch.astype(int)] = 1
                    loss = -np.mean(np.sum(y_onehot * np.log(softmax + 1e-10), axis=1))
                
                epoch_loss += loss
                
                # Backward pass
                grads = self.backward(X_batch, y_batch, output)
                
                # Update parameters
                params = {'weights': self.weights, 'biases': self.biases}
                grad_dict = {f'weights_{i}': g for i, g in enumerate(grads['weights'])}
                grad_dict.update({f'biases_{i}': g for i, g in enumerate(grads['biases'])})
                
                # Simple gradient descent update
                for i in range(len(self.weights)):
                    self.weights[i] -= learning_rate * grads['weights'][i]
                    self.biases[i] -= learning_rate * grads['biases'][i]
            
            if epoch % 10 == 0:
                avg_loss = epoch_loss / (n_samples / batch_size)
                logger.info(f"[NeuralNetwork] Epoch {epoch}, Loss: {avg_loss:.4f}")
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """
        Predict using neural network
        
        Parameters
        ----------
        X : array
            Input data
            
        Returns
        -------
        predictions : array
            Predictions
        """
        output = self.forward(X, training=False)
        
        if output.shape[1] == 1:
            return output.ravel()
        else:
            # Classification - return class with highest probability
            return np.argmax(output, axis=1)
