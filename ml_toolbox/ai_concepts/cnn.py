"""
Convolutional Neural Networks (CNNs)

Implements:
- Convolutional Layers
- Pooling Layers
- Full CNN Architecture
"""
import numpy as np
from typing import Tuple, Optional
import logging

logger = logging.getLogger(__name__)


class ConvLayer:
    """
    Convolutional Layer
    
    Applies convolution operation
    """
    
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int = 3,
                 stride: int = 1, padding: int = 0):
        """
        Initialize convolutional layer
        
        Parameters
        ----------
        in_channels : int
            Number of input channels
        out_channels : int
            Number of output channels
        kernel_size : int
            Kernel size
        stride : int
            Stride
        padding : int
            Padding
        """
        self.in_channels = in_channels
        self.out_channels = out_channels
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding
        
        # Initialize weights (Xavier initialization)
        self.weights = np.random.randn(out_channels, in_channels, kernel_size, kernel_size) * \
                      np.sqrt(2.0 / (in_channels * kernel_size * kernel_size))
        self.bias = np.zeros(out_channels)
    
    def _pad(self, x: np.ndarray) -> np.ndarray:
        """Add padding to input"""
        if self.padding == 0:
            return x
        
        if x.ndim == 3:
            # (channels, height, width)
            return np.pad(x, ((0, 0), (self.padding, self.padding), 
                            (self.padding, self.padding)), mode='constant')
        else:
            # (batch, channels, height, width)
            return np.pad(x, ((0, 0), (0, 0), (self.padding, self.padding),
                            (self.padding, self.padding)), mode='constant')
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Parameters
        ----------
        x : array, shape (batch, channels, height, width) or (channels, height, width)
            Input feature map
            
        Returns
        -------
        output : array
            Convolved output
        """
        # Add batch dimension if needed
        if x.ndim == 3:
            x = x[np.newaxis, :]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, in_channels, in_height, in_width = x.shape
        
        # Pad input
        x_padded = self._pad(x)
        
        # Calculate output dimensions
        out_height = (in_height + 2 * self.padding - self.kernel_size) // self.stride + 1
        out_width = (in_width + 2 * self.padding - self.kernel_size) // self.stride + 1
        
        # Initialize output
        output = np.zeros((batch_size, self.out_channels, out_height, out_width))
        
        # Convolution
        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        # Extract patch
                        patch = x_padded[b, :, h_start:h_end, w_start:w_end]
                        
                        # Convolve
                        output[b, out_c, h, w] = np.sum(patch * self.weights[out_c]) + self.bias[out_c]
        
        if squeeze_output:
            output = output[0]
        
        return output
    
    def backward(self, grad_output: np.ndarray, x: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
        """
        Backward pass
        
        Parameters
        ----------
        grad_output : array
            Gradient from next layer
        x : array
            Input
            
        Returns
        -------
        grad_input : array
            Gradient to previous layer
        grad_weights : array
            Gradient w.r.t. weights
        grad_bias : array
            Gradient w.r.t. bias
        """
        # Simplified backward pass
        # In practice, use automatic differentiation
        
        if x.ndim == 3:
            x = x[np.newaxis, :]
        
        x_padded = self._pad(x)
        batch_size = x.shape[0]
        
        grad_weights = np.zeros_like(self.weights)
        grad_bias = np.sum(grad_output, axis=(0, 2, 3))
        grad_input = np.zeros_like(x_padded)
        
        # Compute gradients (simplified)
        for b in range(batch_size):
            for out_c in range(self.out_channels):
                for h in range(grad_output.shape[2]):
                    for w in range(grad_output.shape[3]):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.kernel_size
                        w_end = w_start + self.kernel_size
                        
                        grad = grad_output[b, out_c, h, w]
                        grad_input[b, :, h_start:h_end, w_start:w_end] += \
                            grad * self.weights[out_c]
                        grad_weights[out_c] += grad * x_padded[b, :, h_start:h_end, w_start:w_end]
        
        # Remove padding from grad_input
        if self.padding > 0:
            grad_input = grad_input[:, :, self.padding:-self.padding, self.padding:-self.padding]
        
        return grad_input, grad_weights, grad_bias


class PoolingLayer:
    """
    Pooling Layer
    
    Max pooling or average pooling
    """
    
    def __init__(self, pool_size: int = 2, stride: int = 2, mode: str = 'max'):
        """
        Initialize pooling layer
        
        Parameters
        ----------
        pool_size : int
            Pooling window size
        stride : int
            Stride
        mode : str
            Pooling mode ('max' or 'avg')
        """
        self.pool_size = pool_size
        self.stride = stride
        self.mode = mode
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Parameters
        ----------
        x : array, shape (batch, channels, height, width) or (channels, height, width)
            Input feature map
            
        Returns
        -------
        output : array
            Pooled output
        """
        if x.ndim == 3:
            x = x[np.newaxis, :]
            squeeze_output = True
        else:
            squeeze_output = False
        
        batch_size, channels, in_height, in_width = x.shape
        
        # Calculate output dimensions
        out_height = (in_height - self.pool_size) // self.stride + 1
        out_width = (in_width - self.pool_size) // self.stride + 1
        
        output = np.zeros((batch_size, channels, out_height, out_width))
        
        for b in range(batch_size):
            for c in range(channels):
                for h in range(out_height):
                    for w in range(out_width):
                        h_start = h * self.stride
                        w_start = w * self.stride
                        h_end = h_start + self.pool_size
                        w_end = w_start + self.pool_size
                        
                        patch = x[b, c, h_start:h_end, w_start:w_end]
                        
                        if self.mode == 'max':
                            output[b, c, h, w] = np.max(patch)
                        else:  # avg
                            output[b, c, h, w] = np.mean(patch)
        
        if squeeze_output:
            output = output[0]
        
        return output


class CNN:
    """
    Convolutional Neural Network
    
    Simple CNN for image classification
    """
    
    def __init__(self, input_shape: Tuple[int, int, int], n_classes: int):
        """
        Initialize CNN
        
        Parameters
        ----------
        input_shape : tuple
            Input shape (channels, height, width)
        n_classes : int
            Number of output classes
        """
        self.input_shape = input_shape
        self.n_classes = n_classes
        
        # Build network
        self.conv1 = ConvLayer(input_shape[0], 32, kernel_size=3, padding=1)
        self.pool1 = PoolingLayer(pool_size=2, stride=2)
        self.conv2 = ConvLayer(32, 64, kernel_size=3, padding=1)
        self.pool2 = PoolingLayer(pool_size=2, stride=2)
        
        # Calculate flattened size (simplified)
        # In practice, calculate from input shape
        self.flatten_size = 64 * (input_shape[1] // 4) * (input_shape[2] // 4)
        
        # Fully connected layers
        self.fc1_weights = np.random.randn(self.flatten_size, 128) * 0.01
        self.fc1_bias = np.zeros(128)
        self.fc2_weights = np.random.randn(128, n_classes) * 0.01
        self.fc2_bias = np.zeros(n_classes)
    
    def forward(self, x: np.ndarray) -> np.ndarray:
        """
        Forward pass
        
        Parameters
        ----------
        x : array, shape (batch, channels, height, width)
            Input images
            
        Returns
        -------
        output : array
            Class logits
        """
        if x.ndim == 3:
            x = x[np.newaxis, :]
        
        # Convolutional layers
        x = self.conv1.forward(x)
        x = np.maximum(0, x)  # ReLU
        x = self.pool1.forward(x)
        
        x = self.conv2.forward(x)
        x = np.maximum(0, x)  # ReLU
        x = self.pool2.forward(x)
        
        # Flatten
        batch_size = x.shape[0]
        x = x.reshape(batch_size, -1)
        
        # Fully connected layers
        x = x @ self.fc1_weights + self.fc1_bias
        x = np.maximum(0, x)  # ReLU
        
        x = x @ self.fc2_weights + self.fc2_bias
        
        return x
    
    def predict(self, x: np.ndarray) -> np.ndarray:
        """
        Predict class labels
        
        Parameters
        ----------
        x : array
            Input images
            
        Returns
        -------
        predictions : array
            Predicted class labels
        """
        logits = self.forward(x)
        return np.argmax(logits, axis=-1)
