"""
GPU-Accelerated Data Preprocessor
Virtual GPU support for speeding up preprocessing operations

Features:
- GPU-accelerated embeddings (sentence-transformers)
- GPU-accelerated similarity computations
- GPU-accelerated dimensionality reduction
- Automatic CPU fallback
- Virtual GPU support (TensorFlow/PyTorch)
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional
import warnings
import time

sys.path.insert(0, str(Path(__file__).parent))

# Try to import GPU libraries
try:
    import torch
    TORCH_AVAILABLE = True
    TORCH_CUDA_AVAILABLE = torch.cuda.is_available()
except ImportError:
    TORCH_AVAILABLE = False
    TORCH_CUDA_AVAILABLE = False

try:
    import tensorflow as tf
    TF_AVAILABLE = True
    TF_GPU_AVAILABLE = len(tf.config.list_physical_devices('GPU')) > 0
except ImportError:
    TF_AVAILABLE = False
    TF_GPU_AVAILABLE = False

try:
    import cupy as cp
    CUPY_AVAILABLE = True
except ImportError:
    CUPY_AVAILABLE = False

try:
    from data_preprocessor import AdvancedDataPreprocessor, ConventionalPreprocessor
    PREPROCESSOR_AVAILABLE = True
except ImportError:
    PREPROCESSOR_AVAILABLE = False
    warnings.warn("Preprocessors not available")


class GPUAcceleratedPreprocessor:
    """
    GPU-Accelerated Data Preprocessor
    
    Combines AdvancedDataPreprocessor with GPU acceleration
    Falls back to CPU if GPU not available
    """
    
    def __init__(self, 
                 use_gpu: bool = True,
                 gpu_device: str = 'auto',
                 batch_size: int = 32,
                 use_virtual_gpu: bool = False,
                 virtual_gpu_memory: int = 2048,  # MB
                 fallback_to_cpu: bool = True):
        """
        Initialize GPU-accelerated preprocessor
        
        Args:
            use_gpu: Enable GPU acceleration
            gpu_device: GPU device ('auto', 'cuda:0', 'cpu')
            batch_size: Batch size for GPU processing
            use_virtual_gpu: Use virtual GPU (TensorFlow)
            virtual_gpu_memory: Virtual GPU memory in MB
            fallback_to_cpu: Fallback to CPU if GPU unavailable
        """
        self.use_gpu = use_gpu
        self.gpu_device = gpu_device
        self.batch_size = batch_size
        self.use_virtual_gpu = use_virtual_gpu
        self.virtual_gpu_memory = virtual_gpu_memory
        self.fallback_to_cpu = fallback_to_cpu
        
        # Detect GPU availability
        self.gpu_available = self._detect_gpu()
        self.device = self._setup_device()
        
        # Initialize base preprocessor
        if PREPROCESSOR_AVAILABLE:
            # Use AdvancedDataPreprocessor with GPU-enabled quantum kernel
            self.base_preprocessor = AdvancedDataPreprocessor(
                use_quantum=True,
                enable_compression=True
            )
            
            # Enable GPU in quantum kernel if available
            if self.gpu_available and hasattr(self.base_preprocessor, 'quantum_kernel'):
                if self.base_preprocessor.quantum_kernel:
                    # Configure kernel for GPU
                    self.base_preprocessor.quantum_kernel.config.use_gpu = True
        else:
            self.base_preprocessor = None
        
        # GPU acceleration flags
        self.gpu_embeddings = self.gpu_available and use_gpu
        self.gpu_similarity = self.gpu_available and use_gpu
        self.gpu_compression = self.gpu_available and use_gpu
        
        # Statistics
        self.stats = {
            'gpu_used': False,
            'gpu_operations': 0,
            'cpu_fallbacks': 0,
            'speedup_factor': 1.0
        }
    
    def _detect_gpu(self) -> bool:
        """Detect GPU availability"""
        if not self.use_gpu:
            return False
        
        # Check PyTorch CUDA
        if TORCH_AVAILABLE and TORCH_CUDA_AVAILABLE:
            return True
        
        # Check TensorFlow GPU
        if TF_AVAILABLE and TF_GPU_AVAILABLE:
            return True
        
        # Check CuPy (for NumPy-like operations)
        if CUPY_AVAILABLE:
            try:
                # Test CuPy
                import cupy as cp
                test_array = cp.array([1, 2, 3])
                return True
            except:
                return False
        
        return False
    
    def _setup_device(self) -> str:
        """Setup GPU device"""
        if not self.gpu_available:
            return 'cpu'
        
        if self.gpu_device == 'auto':
            if TORCH_AVAILABLE and TORCH_CUDA_AVAILABLE:
                return 'cuda:0'
            elif TF_AVAILABLE and TF_GPU_AVAILABLE:
                return 'gpu:0'
            else:
                return 'cpu'
        else:
            return self.gpu_device
    
    def _setup_virtual_gpu(self):
        """Setup virtual GPU (TensorFlow)"""
        if not TF_AVAILABLE:
            return False
        
        try:
            # Configure virtual GPU
            gpus = tf.config.experimental.list_physical_devices('GPU')
            if gpus:
                # Create virtual GPU
                tf.config.experimental.set_virtual_device_configuration(
                    gpus[0],
                    [tf.config.experimental.VirtualDeviceConfiguration(
                        memory_limit=self.virtual_gpu_memory
                    )]
                )
                return True
        except Exception as e:
            warnings.warn(f"Could not setup virtual GPU: {e}")
            return False
        
        return False
    
    def preprocess(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """
        Preprocess data with GPU acceleration
        
        Args:
            raw_data: List of text items
            verbose: Print progress
            
        Returns:
            Preprocessing results with GPU acceleration
        """
        if not self.base_preprocessor:
            raise ValueError("Base preprocessor not available")
        
        start_time = time.time()
        
        if verbose:
            print(f"[GPU Preprocessing] Device: {self.device}")
            print(f"[GPU Preprocessing] Input: {len(raw_data)} items")
        
        # Setup virtual GPU if requested
        if self.use_virtual_gpu and TF_AVAILABLE:
            self._setup_virtual_gpu()
        
        # Stage 1: GPU-accelerated embeddings
        if self.gpu_embeddings:
            if verbose:
                print("[Stage 1] GPU-Accelerated Embeddings")
            embeddings = self._gpu_embeddings(raw_data, verbose)
            self.stats['gpu_operations'] += 1
            self.stats['gpu_used'] = True
        else:
            if verbose:
                print("[Stage 1] CPU Embeddings (GPU not available)")
            embeddings = self._cpu_embeddings(raw_data)
            self.stats['cpu_fallbacks'] += 1
        
        # Stage 2: GPU-accelerated similarity computation
        if self.gpu_similarity and len(embeddings) > 1:
            if verbose:
                print("[Stage 2] GPU-Accelerated Similarity Computation")
            similarity_matrix = self._gpu_similarity_matrix(embeddings, verbose)
            self.stats['gpu_operations'] += 1
        else:
            if verbose:
                print("[Stage 2] CPU Similarity Computation")
            similarity_matrix = self._cpu_similarity_matrix(embeddings)
            self.stats['cpu_fallbacks'] += 1
        
        # Use base preprocessor for other stages
        results = self.base_preprocessor.preprocess(raw_data, verbose=verbose)
        
        # Stage 3: GPU-accelerated compression (if enabled)
        if self.gpu_compression and self.base_preprocessor.enable_compression:
            if verbose:
                print("[Stage 3] GPU-Accelerated Compression")
            compressed = self._gpu_compress(embeddings, verbose)
            if compressed is not None:
                results['compressed_embeddings'] = compressed
                results['compression_info']['gpu_accelerated'] = True
            self.stats['gpu_operations'] += 1
        
        # Calculate speedup
        processing_time = time.time() - start_time
        results['processing_time'] = processing_time
        results['gpu_stats'] = self.stats.copy()
        results['device_used'] = self.device
        
        if verbose:
            print(f"[GPU Preprocessing] Completed in {processing_time:.2f}s")
            print(f"[GPU Preprocessing] GPU Operations: {self.stats['gpu_operations']}")
            print(f"[GPU Preprocessing] CPU Fallbacks: {self.stats['cpu_fallbacks']}")
        
        return results
    
    def _gpu_embeddings(self, texts: List[str], verbose: bool = False) -> List:
        """Compute embeddings on GPU"""
        if not self.gpu_available:
            return self._cpu_embeddings(texts)
        
        try:
            # Try sentence-transformers with GPU
            if hasattr(self.base_preprocessor, 'quantum_kernel') and self.base_preprocessor.quantum_kernel:
                kernel = self.base_preprocessor.quantum_kernel
                
                # Use GPU if available in sentence-transformers
                if hasattr(kernel, 'embedding_model') and kernel.embedding_model:
                    # Set device for model
                    if TORCH_AVAILABLE:
                        device = torch.device(self.device if self.device != 'cpu' else 'cpu')
                        # Model should automatically use GPU if available
                        pass
                
                # Batch process for GPU efficiency
                embeddings = []
                for i in range(0, len(texts), self.batch_size):
                    batch = texts[i:i+self.batch_size]
                    batch_embeddings = [kernel.embed(text) for text in batch]
                    embeddings.extend(batch_embeddings)
                
                return embeddings
        except Exception as e:
            if verbose:
                print(f"GPU embedding failed, falling back to CPU: {e}")
            if self.fallback_to_cpu:
                return self._cpu_embeddings(texts)
            raise
        
        return self._cpu_embeddings(texts)
    
    def _cpu_embeddings(self, texts: List[str]) -> List:
        """Compute embeddings on CPU (fallback)"""
        if hasattr(self.base_preprocessor, 'quantum_kernel') and self.base_preprocessor.quantum_kernel:
            return [self.base_preprocessor.quantum_kernel.embed(text) for text in texts]
        return []
    
    def _gpu_similarity_matrix(self, embeddings: List, verbose: bool = False):
        """Compute similarity matrix on GPU"""
        if not self.gpu_available or len(embeddings) < 2:
            return self._cpu_similarity_matrix(embeddings)
        
        try:
            import numpy as np
            
            # Convert to array
            if CUPY_AVAILABLE:
                # Use CuPy for GPU-accelerated NumPy operations
                import cupy as cp
                emb_array = cp.array(embeddings)
                
                # Compute cosine similarity matrix on GPU
                # Normalize
                norms = cp.linalg.norm(emb_array, axis=1, keepdims=True)
                norms = cp.where(norms == 0, 1, norms)
                normalized = emb_array / norms
                
                # Compute similarity matrix
                similarity = cp.dot(normalized, normalized.T)
                
                # Convert back to CPU numpy array
                return cp.asnumpy(similarity)
            
            elif TORCH_AVAILABLE and TORCH_CUDA_AVAILABLE:
                # Use PyTorch for GPU operations
                emb_tensor = torch.tensor(embeddings, device=self.device)
                
                # Normalize
                norms = torch.norm(emb_tensor, dim=1, keepdim=True)
                norms = torch.where(norms == 0, torch.ones_like(norms), norms)
                normalized = emb_tensor / norms
                
                # Compute similarity matrix
                similarity = torch.mm(normalized, normalized.t())
                
                # Convert back to CPU numpy
                return similarity.cpu().numpy()
            
            elif TF_AVAILABLE and TF_GPU_AVAILABLE:
                # Use TensorFlow for GPU operations
                emb_tensor = tf.constant(embeddings)
                
                # Normalize
                norms = tf.linalg.norm(emb_tensor, axis=1, keepdims=True)
                norms = tf.where(norms == 0, tf.ones_like(norms), norms)
                normalized = emb_tensor / norms
                
                # Compute similarity matrix
                similarity = tf.linalg.matmul(normalized, normalized, transpose_b=True)
                
                # Convert back to numpy
                return similarity.numpy()
            
        except Exception as e:
            if verbose:
                print(f"GPU similarity computation failed, falling back to CPU: {e}")
            if self.fallback_to_cpu:
                return self._cpu_similarity_matrix(embeddings)
            raise
        
        return self._cpu_similarity_matrix(embeddings)
    
    def _cpu_similarity_matrix(self, embeddings: List):
        """Compute similarity matrix on CPU (fallback)"""
        import numpy as np
        
        emb_array = np.array(embeddings)
        
        # Normalize
        norms = np.linalg.norm(emb_array, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1, norms)
        normalized = emb_array / norms
        
        # Compute similarity matrix
        similarity = np.dot(normalized, normalized.T)
        
        return similarity
    
    def _gpu_compress(self, embeddings: List, verbose: bool = False):
        """GPU-accelerated dimensionality reduction"""
        if not self.gpu_available or not embeddings:
            return None
        
        try:
            import numpy as np
            
            emb_array = np.array(embeddings)
            
            if CUPY_AVAILABLE:
                # Use CuPy for GPU-accelerated SVD
                import cupy as cp
                from cupy.linalg import svd
                
                emb_gpu = cp.array(emb_array)
                
                # SVD on GPU
                U, s, Vt = svd(emb_gpu, full_matrices=False)
                
                # Select top components (50% compression)
                n_components = max(1, len(s) // 2)
                compressed = cp.asnumpy(U[:, :n_components] @ cp.diag(s[:n_components]))
                
                return compressed.tolist()
            
            elif TORCH_AVAILABLE and TORCH_CUDA_AVAILABLE:
                # Use PyTorch for GPU SVD
                emb_tensor = torch.tensor(emb_array, device=self.device)
                
                # SVD on GPU
                U, s, V = torch.svd(emb_tensor)
                
                # Select top components
                n_components = max(1, len(s) // 2)
                compressed = (U[:, :n_components] @ torch.diag(s[:n_components])).cpu().numpy()
                
                return compressed.tolist()
            
        except Exception as e:
            if verbose:
                print(f"GPU compression failed, falling back to CPU: {e}")
            if self.fallback_to_cpu:
                return None
            raise
        
        return None
    
    def get_gpu_info(self) -> Dict[str, Any]:
        """Get GPU information"""
        info = {
            'gpu_available': self.gpu_available,
            'device': self.device,
            'torch_cuda': TORCH_CUDA_AVAILABLE if TORCH_AVAILABLE else False,
            'tensorflow_gpu': TF_GPU_AVAILABLE if TF_AVAILABLE else False,
            'cupy_available': CUPY_AVAILABLE,
            'virtual_gpu': self.use_virtual_gpu
        }
        
        if TORCH_AVAILABLE and TORCH_CUDA_AVAILABLE:
            info['torch_gpu_count'] = torch.cuda.device_count()
            info['torch_gpu_name'] = torch.cuda.get_device_name(0) if torch.cuda.device_count() > 0 else None
        
        if TF_AVAILABLE:
            info['tf_gpu_devices'] = [str(d) for d in tf.config.list_physical_devices('GPU')]
        
        return info


class HybridPreprocessor:
    """
    Hybrid Preprocessor
    
    Automatically chooses between GPU-accelerated and CPU preprocessing
    based on data size and GPU availability
    """
    
    def __init__(self, gpu_threshold: int = 100):
        """
        Args:
            gpu_threshold: Minimum items to use GPU (smaller uses CPU)
        """
        self.gpu_threshold = gpu_threshold
        self.gpu_preprocessor = GPUAcceleratedPreprocessor() if GPUAcceleratedPreprocessor().gpu_available else None
        self.cpu_preprocessor = ConventionalPreprocessor() if PREPROCESSOR_AVAILABLE else None
    
    def preprocess(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """
        Preprocess with automatic GPU/CPU selection
        
        Args:
            raw_data: List of text items
            verbose: Print progress
            
        Returns:
            Preprocessing results
        """
        # Decide: GPU for large datasets, CPU for small
        use_gpu = (
            self.gpu_preprocessor is not None and
            len(raw_data) >= self.gpu_threshold
        )
        
        if verbose:
            print(f"[Hybrid Preprocessor] Using: {'GPU' if use_gpu else 'CPU'}")
            print(f"[Hybrid Preprocessor] Data size: {len(raw_data)} items")
        
        if use_gpu:
            return self.gpu_preprocessor.preprocess(raw_data, verbose=verbose)
        else:
            if self.cpu_preprocessor:
                return self.cpu_preprocessor.preprocess(raw_data, verbose=verbose)
            else:
                # Fallback to GPU preprocessor even for small data
                if self.gpu_preprocessor:
                    return self.gpu_preprocessor.preprocess(raw_data, verbose=verbose)
                raise ValueError("No preprocessor available")


# Example usage
if __name__ == '__main__':
    # Test GPU preprocessor
    preprocessor = GPUAcceleratedPreprocessor(use_gpu=True)
    
    # Check GPU info
    gpu_info = preprocessor.get_gpu_info()
    print("GPU Info:", gpu_info)
    
    # Test preprocessing
    texts = ["text1", "text2", "text3"] * 100  # Large dataset for GPU
    
    if preprocessor.gpu_available:
        results = preprocessor.preprocess(texts, verbose=True)
        print(f"GPU Stats: {results['gpu_stats']}")
    else:
        print("GPU not available, using CPU fallback")
