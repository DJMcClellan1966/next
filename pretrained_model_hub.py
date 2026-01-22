"""
Pre-trained Model Hub
Model repository with transfer learning and fine-tuning capabilities

Features:
- Model repository (like Hugging Face Hub)
- Pre-trained models (CNNs, Transformers, etc.)
- Model sharing and discovery
- Transfer learning utilities
- Fine-tuning pipelines
- Model evaluation benchmarks
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable
import json
import datetime
import pickle
import hashlib
import warnings
import urllib.request
import urllib.parse

sys.path.insert(0, str(Path(__file__).parent))


class PretrainedModel:
    """Pre-trained model metadata"""
    
    def __init__(
        self,
        model_id: str,
        name: str,
        description: str,
        model_type: str,
        download_url: Optional[str] = None,
        local_path: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ):
        """
        Args:
            model_id: Unique model identifier
            name: Model name
            description: Model description
            model_type: Type (e.g., 'cnn', 'transformer', 'rnn')
            download_url: URL to download model
            local_path: Local path to model file
            metadata: Additional metadata
        """
        self.model_id = model_id
        self.name = name
        self.description = description
        self.model_type = model_type
        self.download_url = download_url
        self.local_path = local_path
        self.metadata = metadata or {}
        self.created_at = datetime.datetime.now()
        self.download_count = 0
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return {
            'model_id': self.model_id,
            'name': self.name,
            'description': self.description,
            'model_type': self.model_type,
            'download_url': self.download_url,
            'local_path': self.local_path,
            'metadata': self.metadata,
            'created_at': self.created_at.isoformat(),
            'download_count': self.download_count
        }


class PretrainedModelHub:
    """
    Pre-trained Model Hub
    
    Model repository with transfer learning capabilities
    """
    
    def __init__(self, hub_path: str = "model_hub"):
        """
        Args:
            hub_path: Path to store models and metadata
        """
        self.hub_path = Path(hub_path)
        self.hub_path.mkdir(parents=True, exist_ok=True)
        self.metadata_file = self.hub_path / "models.json"
        self.models: Dict[str, PretrainedModel] = {}
        self._check_dependencies()
        self._load_hub()
        self._initialize_default_models()
    
    def _check_dependencies(self):
        """Check if required dependencies are available"""
        try:
            import torch
            self.torch_available = True
        except ImportError:
            self.torch_available = False
            warnings.warn("PyTorch not available. Some features will be limited.")
        
        try:
            from transformers import AutoModel, AutoTokenizer
            self.transformers_available = True
        except ImportError:
            self.transformers_available = False
            warnings.warn("Transformers not available. Install with: pip install transformers")
    
    def _load_hub(self):
        """Load hub metadata"""
        try:
            if self.metadata_file.exists():
                with open(self.metadata_file, 'r') as f:
                    data = json.load(f)
                    for model_id, model_data in data.items():
                        model = PretrainedModel(
                            model_id=model_data['model_id'],
                            name=model_data['name'],
                            description=model_data['description'],
                            model_type=model_data['model_type'],
                            download_url=model_data.get('download_url'),
                            local_path=model_data.get('local_path'),
                            metadata=model_data.get('metadata', {})
                        )
                        model.created_at = datetime.datetime.fromisoformat(model_data['created_at'])
                        model.download_count = model_data.get('download_count', 0)
                        self.models[model_id] = model
        except Exception as e:
            print(f"Error loading hub: {e}")
            self.models = {}
    
    def _save_hub(self):
        """Save hub metadata"""
        try:
            data = {model_id: model.to_dict() 
                   for model_id, model in self.models.items()}
            with open(self.metadata_file, 'w') as f:
                json.dump(data, f, indent=2, default=str)
        except Exception as e:
            print(f"Error saving hub: {e}")
    
    def _initialize_default_models(self):
        """Initialize default pre-trained models"""
        if len(self.models) > 0:
            return  # Already initialized
        
        default_models = [
            {
                'model_id': 'resnet18-imagenet',
                'name': 'ResNet-18 (ImageNet)',
                'description': 'ResNet-18 pre-trained on ImageNet',
                'model_type': 'cnn',
                'download_url': 'pytorch/vision:resnet18',
                'metadata': {
                    'dataset': 'ImageNet',
                    'accuracy': '69.8%',
                    'parameters': '11.7M'
                }
            },
            {
                'model_id': 'bert-base-uncased',
                'name': 'BERT Base Uncased',
                'description': 'BERT base model (uncased)',
                'model_type': 'transformer',
                'download_url': 'bert-base-uncased',
                'metadata': {
                    'vocab_size': '30522',
                    'hidden_size': '768',
                    'num_layers': '12'
                }
            },
            {
                'model_id': 'distilbert-base-uncased',
                'name': 'DistilBERT Base Uncased',
                'description': 'DistilBERT base model (uncased) - smaller, faster',
                'model_type': 'transformer',
                'download_url': 'distilbert-base-uncased',
                'metadata': {
                    'vocab_size': '30522',
                    'hidden_size': '768',
                    'num_layers': '6'
                }
            }
        ]
        
        for model_data in default_models:
            if model_data['model_id'] not in self.models:
                model = PretrainedModel(**model_data)
                self.models[model_data['model_id']] = model
        
        self._save_hub()
    
    def register_model(
        self,
        model_id: str,
        name: str,
        description: str,
        model_type: str,
        model: Any,
        download_url: Optional[str] = None,
        metadata: Optional[Dict[str, Any]] = None
    ) -> bool:
        """
        Register a pre-trained model
        
        Args:
            model_id: Unique model identifier
            name: Model name
            description: Model description
            model_type: Type (e.g., 'cnn', 'transformer')
            model: Model object
            download_url: URL to download (optional)
            metadata: Additional metadata
            
        Returns:
            True if successful
        """
        # Save model locally
        model_dir = self.hub_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        
        try:
            with open(model_path, 'wb') as f:
                pickle.dump(model, f)
        except Exception as e:
            print(f"Error saving model: {e}")
            return False
        
        # Register model
        pretrained_model = PretrainedModel(
            model_id=model_id,
            name=name,
            description=description,
            model_type=model_type,
            download_url=download_url,
            local_path=str(model_path),
            metadata=metadata or {}
        )
        
        self.models[model_id] = pretrained_model
        self._save_hub()
        
        return True
    
    def download_model(self, model_id: str, force: bool = False) -> Optional[Any]:
        """
        Download and load pre-trained model
        
        Args:
            model_id: Model identifier
            force: Force re-download even if exists
            
        Returns:
            Model object or None
        """
        if model_id not in self.models:
            print(f"Model {model_id} not found")
            return None
        
        model_info = self.models[model_id]
        
        # Check if already downloaded
        if model_info.local_path and Path(model_info.local_path).exists() and not force:
            try:
                with open(model_info.local_path, 'rb') as f:
                    model = pickle.load(f)
                model_info.download_count += 1
                self._save_hub()
                return model
            except Exception as e:
                print(f"Error loading model: {e}")
        
        # Download from URL
        if model_info.download_url:
            return self._download_from_url(model_id, model_info.download_url, force)
        
        return None
    
    def _download_from_url(self, model_id: str, url: str, force: bool) -> Optional[Any]:
        """Download model from URL"""
        model_info = self.models[model_id]
        model_dir = self.hub_path / model_id
        model_dir.mkdir(parents=True, exist_ok=True)
        model_path = model_dir / "model.pkl"
        
        # Try Hugging Face transformers
        if self.transformers_available and not url.startswith('http'):
            try:
                from transformers import AutoModel, AutoTokenizer
                
                # Download model
                model = AutoModel.from_pretrained(url)
                tokenizer = AutoTokenizer.from_pretrained(url)
                
                # Save model
                model_dir.mkdir(parents=True, exist_ok=True)
                model.save_pretrained(str(model_dir))
                tokenizer.save_pretrained(str(model_dir))
                
                # Update metadata
                model_info.local_path = str(model_dir)
                model_info.download_count += 1
                self._save_hub()
                
                return {'model': model, 'tokenizer': tokenizer}
            except Exception as e:
                print(f"Error downloading from Hugging Face: {e}")
        
        # Try PyTorch vision models
        if self.torch_available and 'vision' in url:
            try:
                import torchvision.models as models
                
                model_name = url.split(':')[-1]
                if hasattr(models, model_name):
                    model = getattr(models, model_name)(pretrained=True)
                    
                    # Save model
                    with open(model_path, 'wb') as f:
                        pickle.dump(model, f)
                    
                    model_info.local_path = str(model_path)
                    model_info.download_count += 1
                    self._save_hub()
                    
                    return model
            except Exception as e:
                print(f"Error downloading PyTorch model: {e}")
        
        return None
    
    def list_models(
        self,
        model_type: Optional[str] = None,
        search_query: Optional[str] = None
    ) -> List[PretrainedModel]:
        """
        List available models
        
        Args:
            model_type: Filter by type (optional)
            search_query: Search in name/description (optional)
            
        Returns:
            List of models
        """
        models = list(self.models.values())
        
        # Filter by type
        if model_type:
            models = [m for m in models if m.model_type == model_type]
        
        # Search
        if search_query:
            query_lower = search_query.lower()
            models = [
                m for m in models
                if query_lower in m.name.lower() or query_lower in m.description.lower()
            ]
        
        return models
    
    def get_model_info(self, model_id: str) -> Optional[PretrainedModel]:
        """Get model information"""
        return self.models.get(model_id)
    
    def transfer_learning(
        self,
        base_model_id: str,
        num_classes: int,
        freeze_base: bool = True
    ) -> Optional[Any]:
        """
        Create transfer learning model from pre-trained model
        
        Args:
            base_model_id: Base model identifier
            num_classes: Number of output classes
            freeze_base: Whether to freeze base model weights
            
        Returns:
            Transfer learning model or None
        """
        if not self.torch_available:
            warnings.warn("PyTorch required for transfer learning")
            return None
        
        import torch
        import torch.nn as nn
        
        # Load base model
        base_model_data = self.download_model(base_model_id)
        if base_model_data is None:
            return None
        
        # Extract model
        if isinstance(base_model_data, dict):
            base_model = base_model_data.get('model')
        else:
            base_model = base_model_data
        
        if base_model is None:
            return None
        
        # Freeze base model
        if freeze_base:
            for param in base_model.parameters():
                param.requires_grad = False
        
        # Modify last layer for new task
        if hasattr(base_model, 'fc'):  # ResNet
            num_features = base_model.fc.in_features
            base_model.fc = nn.Linear(num_features, num_classes)
        elif hasattr(base_model, 'classifier'):  # Some models
            if isinstance(base_model.classifier, nn.Linear):
                num_features = base_model.classifier.in_features
                base_model.classifier = nn.Linear(num_features, num_classes)
        else:
            # Generic: add classifier
            num_features = base_model.config.hidden_size if hasattr(base_model, 'config') else 768
            base_model.classifier = nn.Linear(num_features, num_classes)
        
        return base_model
    
    def fine_tune_model(
        self,
        model_id: str,
        train_data: Any,
        val_data: Optional[Any] = None,
        num_epochs: int = 3,
        learning_rate: float = 2e-5,
        batch_size: int = 32
    ) -> Optional[Dict[str, Any]]:
        """
        Fine-tune a pre-trained model
        
        Args:
            model_id: Model identifier
            train_data: Training data
            val_data: Validation data (optional)
            num_epochs: Number of training epochs
            learning_rate: Learning rate
            batch_size: Batch size
            
        Returns:
            Fine-tuned model and training history
        """
        if not self.torch_available:
            warnings.warn("PyTorch required for fine-tuning")
            return None
        
        import torch
        import torch.nn as nn
        import torch.optim as optim
        from torch.utils.data import DataLoader
        
        # Load model
        model_data = self.download_model(model_id)
        if model_data is None:
            return None
        
        model = model_data.get('model') if isinstance(model_data, dict) else model_data
        if model is None:
            return None
        
        # Setup training
        device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        model = model.to(device)
        
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.Adam(model.parameters(), lr=learning_rate)
        
        train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
        val_loader = DataLoader(val_data, batch_size=batch_size) if val_data else None
        
        # Training loop
        history = {'loss': [], 'accuracy': []}
        
        for epoch in range(num_epochs):
            model.train()
            epoch_loss = 0.0
            correct = 0
            total = 0
            
            for batch_idx, (data, target) in enumerate(train_loader):
                data, target = data.to(device), target.to(device)
                
                optimizer.zero_grad()
                output = model(data)
                loss = criterion(output, target)
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                _, predicted = output.max(1)
                total += target.size(0)
                correct += predicted.eq(target).sum().item()
            
            avg_loss = epoch_loss / len(train_loader)
            accuracy = 100. * correct / total
            history['loss'].append(avg_loss)
            history['accuracy'].append(accuracy)
            
            print(f"Epoch {epoch+1}/{num_epochs} - Loss: {avg_loss:.4f}, Accuracy: {accuracy:.2f}%")
        
        return {
            'model': model,
            'history': history,
            'final_accuracy': history['accuracy'][-1] if history['accuracy'] else 0
        }
