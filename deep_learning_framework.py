"""
Deep Learning Framework
Comprehensive deep learning capabilities for ML Toolbox

Features:
- Advanced neural network architectures (CNN, RNN, LSTM, Transformer)
- Training utilities and callbacks
- Advanced optimizers and learning rate schedules
- Regularization techniques
- Transfer learning support
- Model architectures (ResNet, VGG, etc.)
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
import numpy as np
import warnings

sys.path.insert(0, str(Path(__file__).parent))


class DeepLearningFramework:
    """
    Comprehensive Deep Learning Framework
    
    Advanced neural network architectures and training utilities
    """
    
    def __init__(self):
        """Initialize deep learning framework"""
        self.dependencies = ['torch', 'torchvision', 'numpy']
        self._check_dependencies()
    
    def _check_dependencies(self):
        """Check if PyTorch is available"""
        try:
            import torch
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            warnings.warn("PyTorch not available. Deep learning features will be limited.")
    
    def create_cnn(
        self,
        input_channels: int = 3,
        num_classes: int = 10,
        architecture: str = 'simple'
    ) -> Any:
        """
        Create Convolutional Neural Network
        
        Args:
            input_channels: Number of input channels
            num_classes: Number of output classes
            architecture: Architecture type ('simple', 'vgg', 'resnet')
            
        Returns:
            CNN model
        """
        if not self.torch_available:
            return {'error': 'PyTorch required for CNN. Install with: pip install torch'}
        
        import torch
        import torch.nn as nn
        
        if architecture == 'simple':
            return self._create_simple_cnn(input_channels, num_classes)
        elif architecture == 'vgg':
            return self._create_vgg_like(input_channels, num_classes)
        elif architecture == 'resnet':
            return self._create_resnet_like(input_channels, num_classes)
        else:
            return self._create_simple_cnn(input_channels, num_classes)
    
    def _create_simple_cnn(self, input_channels: int, num_classes: int):
        """Create simple CNN"""
        import torch.nn as nn
        
        class SimpleCNN(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(input_channels, 32, kernel_size=3, padding=1)
                self.conv2 = nn.Conv2d(32, 64, kernel_size=3, padding=1)
                self.conv3 = nn.Conv2d(64, 128, kernel_size=3, padding=1)
                self.pool = nn.MaxPool2d(2, 2)
                self.dropout = nn.Dropout(0.5)
                self.fc1 = nn.Linear(128 * 4 * 4, 512)
                self.fc2 = nn.Linear(512, num_classes)
                self.relu = nn.ReLU()
            
            def forward(self, x):
                x = self.pool(self.relu(self.conv1(x)))
                x = self.pool(self.relu(self.conv2(x)))
                x = self.pool(self.relu(self.conv3(x)))
                x = x.view(-1, 128 * 4 * 4)
                x = self.dropout(self.relu(self.fc1(x)))
                x = self.fc2(x)
                return x
        
        return SimpleCNN(input_channels, num_classes)
    
    def _create_vgg_like(self, input_channels: int, num_classes: int):
        """Create VGG-like architecture"""
        import torch.nn as nn
        
        class VGGLike(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()
                self.features = nn.Sequential(
                    nn.Conv2d(input_channels, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(64, 64, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(64, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(128, 128, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                    nn.Conv2d(128, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.Conv2d(256, 256, 3, padding=1),
                    nn.ReLU(inplace=True),
                    nn.MaxPool2d(2, 2),
                )
                self.classifier = nn.Sequential(
                    nn.Dropout(0.5),
                    nn.Linear(256 * 4 * 4, 4096),
                    nn.ReLU(inplace=True),
                    nn.Dropout(0.5),
                    nn.Linear(4096, 4096),
                    nn.ReLU(inplace=True),
                    nn.Linear(4096, num_classes),
                )
            
            def forward(self, x):
                x = self.features(x)
                x = x.view(x.size(0), -1)
                x = self.classifier(x)
                return x
        
        return VGGLike(input_channels, num_classes)
    
    def _create_resnet_like(self, input_channels: int, num_classes: int):
        """Create ResNet-like architecture"""
        import torch.nn as nn
        
        class ResidualBlock(nn.Module):
            def __init__(self, in_channels, out_channels, stride=1):
                super().__init__()
                self.conv1 = nn.Conv2d(in_channels, out_channels, 3, stride, 1, bias=False)
                self.bn1 = nn.BatchNorm2d(out_channels)
                self.conv2 = nn.Conv2d(out_channels, out_channels, 3, 1, 1, bias=False)
                self.bn2 = nn.BatchNorm2d(out_channels)
                self.relu = nn.ReLU(inplace=True)
                
                self.shortcut = nn.Sequential()
                if stride != 1 or in_channels != out_channels:
                    self.shortcut = nn.Sequential(
                        nn.Conv2d(in_channels, out_channels, 1, stride, bias=False),
                        nn.BatchNorm2d(out_channels)
                    )
            
            def forward(self, x):
                out = self.relu(self.bn1(self.conv1(x)))
                out = self.bn2(self.conv2(out))
                out += self.shortcut(x)
                out = self.relu(out)
                return out
        
        class ResNetLike(nn.Module):
            def __init__(self, input_channels, num_classes):
                super().__init__()
                self.conv1 = nn.Conv2d(input_channels, 64, 7, 2, 3, bias=False)
                self.bn1 = nn.BatchNorm2d(64)
                self.relu = nn.ReLU(inplace=True)
                self.maxpool = nn.MaxPool2d(3, 2, 1)
                
                self.layer1 = self._make_layer(64, 64, 2)
                self.layer2 = self._make_layer(64, 128, 2, stride=2)
                self.layer3 = self._make_layer(128, 256, 2, stride=2)
                
                self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
                self.fc = nn.Linear(256, num_classes)
            
            def _make_layer(self, in_channels, out_channels, blocks, stride=1):
                layers = [ResidualBlock(in_channels, out_channels, stride)]
                for _ in range(1, blocks):
                    layers.append(ResidualBlock(out_channels, out_channels))
                return nn.Sequential(*layers)
            
            def forward(self, x):
                x = self.relu(self.bn1(self.conv1(x)))
                x = self.maxpool(x)
                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)
                x = self.avgpool(x)
                x = x.view(x.size(0), -1)
                x = self.fc(x)
                return x
        
        return ResNetLike(input_channels, num_classes)
    
    def create_rnn(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int = 1,
        num_classes: int = 10,
        rnn_type: str = 'LSTM'
    ) -> Any:
        """
        Create Recurrent Neural Network
        
        Args:
            input_size: Input feature size
            hidden_size: Hidden state size
            num_layers: Number of RNN layers
            num_classes: Number of output classes
            rnn_type: Type of RNN ('RNN', 'LSTM', 'GRU')
            
        Returns:
            RNN model
        """
        if not self.torch_available:
            return {'error': 'PyTorch required for RNN. Install with: pip install torch'}
        
        import torch
        import torch.nn as nn
        
        class RNNModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, num_classes, rnn_type):
                super().__init__()
                self.hidden_size = hidden_size
                self.num_layers = num_layers
                
                if rnn_type == 'LSTM':
                    self.rnn = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
                elif rnn_type == 'GRU':
                    self.rnn = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
                else:
                    self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
                
                self.fc = nn.Linear(hidden_size, num_classes)
                self.dropout = nn.Dropout(0.3)
            
            def forward(self, x):
                h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                if isinstance(self.rnn, nn.LSTM):
                    c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
                    out, _ = self.rnn(x, (h0, c0))
                else:
                    out, _ = self.rnn(x, h0)
                
                out = out[:, -1, :]  # Take last output
                out = self.dropout(out)
                out = self.fc(out)
                return out
        
        return RNNModel(input_size, hidden_size, num_layers, num_classes, rnn_type)
    
    def create_transformer(
        self,
        vocab_size: int,
        d_model: int = 512,
        nhead: int = 8,
        num_layers: int = 6,
        dim_feedforward: int = 2048,
        max_seq_length: int = 512,
        num_classes: int = 10
    ) -> Any:
        """
        Create Transformer model
        
        Args:
            vocab_size: Vocabulary size
            d_model: Model dimension
            nhead: Number of attention heads
            num_layers: Number of transformer layers
            dim_feedforward: Feedforward dimension
            max_seq_length: Maximum sequence length
            num_classes: Number of output classes
            
        Returns:
            Transformer model
        """
        if not self.torch_available:
            return {'error': 'PyTorch required for Transformer. Install with: pip install torch'}
        
        import torch
        import torch.nn as nn
        
        class TransformerModel(nn.Module):
            def __init__(self, vocab_size, d_model, nhead, num_layers, 
                        dim_feedforward, max_seq_length, num_classes):
                super().__init__()
                self.embedding = nn.Embedding(vocab_size, d_model)
                self.pos_encoding = nn.Parameter(
                    torch.randn(1, max_seq_length, d_model)
                )
                
                encoder_layer = nn.TransformerEncoderLayer(
                    d_model, nhead, dim_feedforward, dropout=0.1, batch_first=True
                )
                self.transformer = nn.TransformerEncoder(encoder_layer, num_layers)
                
                self.fc = nn.Linear(d_model, num_classes)
                self.dropout = nn.Dropout(0.1)
            
            def forward(self, x):
                x = self.embedding(x) + self.pos_encoding[:, :x.size(1), :]
                x = self.dropout(x)
                x = self.transformer(x)
                x = x.mean(dim=1)  # Global average pooling
                x = self.fc(x)
                return x
        
        return TransformerModel(vocab_size, d_model, nhead, num_layers, 
                               dim_feedforward, max_seq_length, num_classes)
    
    def create_optimizer(
        self,
        model: Any,
        optimizer_type: str = 'Adam',
        learning_rate: float = 0.001,
        **kwargs
    ) -> Any:
        """
        Create optimizer
        
        Args:
            model: PyTorch model
            optimizer_type: Type ('Adam', 'SGD', 'RMSprop', 'AdamW', 'AdaGrad')
            learning_rate: Learning rate
            **kwargs: Additional optimizer parameters
            
        Returns:
            Optimizer
        """
        if not self.torch_available:
            return {'error': 'PyTorch required for optimizer'}
        
        import torch.optim as optim
        
        optimizers = {
            'Adam': optim.Adam,
            'SGD': optim.SGD,
            'RMSprop': optim.RMSprop,
            'AdamW': optim.AdamW,
            'AdaGrad': optim.Adagrad
        }
        
        optimizer_class = optimizers.get(optimizer_type, optim.Adam)
        return optimizer_class(model.parameters(), lr=learning_rate, **kwargs)
    
    def create_lr_scheduler(
        self,
        optimizer: Any,
        scheduler_type: str = 'StepLR',
        **kwargs
    ) -> Any:
        """
        Create learning rate scheduler
        
        Args:
            optimizer: Optimizer
            scheduler_type: Type ('StepLR', 'CosineAnnealing', 'ReduceLROnPlateau', 'ExponentialLR')
            **kwargs: Scheduler parameters
            
        Returns:
            Learning rate scheduler
        """
        if not self.torch_available:
            return {'error': 'PyTorch required for scheduler'}
        
        import torch.optim.lr_scheduler as lr_scheduler
        
        schedulers = {
            'StepLR': lr_scheduler.StepLR,
            'CosineAnnealing': lr_scheduler.CosineAnnealingLR,
            'ReduceLROnPlateau': lr_scheduler.ReduceLROnPlateau,
            'ExponentialLR': lr_scheduler.ExponentialLR,
            'MultiStepLR': lr_scheduler.MultiStepLR
        }
        
        scheduler_class = schedulers.get(scheduler_type, lr_scheduler.StepLR)
        
        if scheduler_type == 'ReduceLROnPlateau':
            return scheduler_class(optimizer, **kwargs)
        else:
            return scheduler_class(optimizer, **kwargs)
    
    def train_model(
        self,
        model: Any,
        train_loader: Any,
        val_loader: Optional[Any] = None,
        num_epochs: int = 10,
        optimizer: Optional[Any] = None,
        criterion: Optional[Any] = None,
        device: str = 'cpu',
        callbacks: Optional[List[Callable]] = None
    ) -> Dict[str, Any]:
        """
        Train deep learning model
        
        Args:
            model: PyTorch model
            train_loader: Training data loader
            val_loader: Validation data loader (optional)
            num_epochs: Number of training epochs
            optimizer: Optimizer (if None, creates Adam)
            criterion: Loss function (if None, creates CrossEntropyLoss)
            device: Device ('cpu' or 'cuda')
            callbacks: List of callback functions
            
        Returns:
            Training history
        """
        if not self.torch_available:
            return {'error': 'PyTorch required for training'}
        
        import torch
        import torch.nn as nn
        
        model = model.to(device)
        
        if optimizer is None:
            optimizer = self.create_optimizer(model, 'Adam', 0.001)
        
        if criterion is None:
            criterion = nn.CrossEntropyLoss()
        
        history = {
            'train_loss': [],
            'train_acc': [],
            'val_loss': [],
            'val_acc': []
        }
        
        for epoch in range(num_epochs):
            # Training phase
            model.train()
            train_loss = 0.0
            train_correct = 0
            train_total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                train_loss += loss.item()
                _, predicted = output.max(1)
                train_total += target.size(0)
                train_correct += predicted.eq(target).sum().item()
            
            train_loss /= len(train_loader)
            train_acc = 100. * train_correct / train_total
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            
            # Validation phase
            if val_loader is not None:
                model.eval()
                val_loss = 0.0
                val_correct = 0
                val_total = 0
                
                with torch.no_grad():
                    for data, target in val_loader:
                        data, target = data.to(device), target.to(device)
                        output = model(data)
                        loss = criterion(output, target)
                        
                        val_loss += loss.item()
                        _, predicted = output.max(1)
                        val_total += target.size(0)
                        val_correct += predicted.eq(target).sum().item()
                
                val_loss /= len(val_loader)
                val_acc = 100. * val_correct / val_total
                history['val_loss'].append(val_loss)
                history['val_acc'].append(val_acc)
            
            # Callbacks
            if callbacks:
                for callback in callbacks:
                    callback(epoch, history)
        
        return history
    
    def evaluate_model(
        self,
        model: Any,
        test_loader: Any,
        device: str = 'cpu'
    ) -> Dict[str, float]:
        """
        Evaluate model
        
        Args:
            model: PyTorch model
            test_loader: Test data loader
            device: Device ('cpu' or 'cuda')
            
        Returns:
            Evaluation metrics
        """
        if not self.torch_available:
            return {'error': 'PyTorch required for evaluation'}
        
        import torch
        import torch.nn as nn
        
        model = model.to(device)
        model.eval()
        
        correct = 0
        total = 0
        test_loss = 0.0
        criterion = nn.CrossEntropyLoss()
        
        with torch.no_grad():
            for data, target in test_loader:
                data, target = data.to(device), target.to(device)
                output = model(data)
                loss = criterion(output, target)
                
                test_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
        
        accuracy = 100. * correct / total
        avg_loss = test_loss / len(test_loader)
        
        return {
            'accuracy': accuracy,
            'loss': avg_loss,
            'correct': correct,
            'total': total
        }
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'torch': 'torch>=2.0.0',
            'torchvision': 'torchvision>=0.15.0',
            'numpy': 'numpy>=1.26.0'
        }
