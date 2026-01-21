"""
Advanced Data Preprocessor
Combines Quantum Kernel + PocketFence Kernel for superior data preprocessing
Includes dimensionality reduction for data compression
"""
import sys
from pathlib import Path
import time
from collections import defaultdict
from typing import List, Dict, Tuple, Any, Optional
import re
import hashlib

sys.path.insert(0, str(Path(__file__).parent))

from quantum_kernel import get_kernel, KernelConfig
import requests
import numpy as np

# Import data scrubbing tools
try:
    from data_scrubbing import DataScrubber, AdvancedDataScrubber
    SCRUBBING_AVAILABLE = True
except ImportError:
    SCRUBBING_AVAILABLE = False
    print("Warning: Data scrubbing tools not available. Install data_scrubbing module.")

# Try to import sklearn for PCA and ML evaluation
try:
    from sklearn.decomposition import PCA, TruncatedSVD
    from sklearn.preprocessing import StandardScaler
    from sklearn.model_selection import (
        train_test_split, cross_val_score, KFold, StratifiedKFold,
        GridSearchCV, RandomizedSearchCV, learning_curve, validation_curve
    )
    from sklearn.metrics import (
        accuracy_score, precision_score, recall_score, f1_score,
        classification_report, confusion_matrix, roc_auc_score,
        mean_squared_error, mean_absolute_error, r2_score
    )
    SKLEARN_AVAILABLE = True
except ImportError:
    SKLEARN_AVAILABLE = False
    print("Warning: sklearn not available. Dimensionality reduction and ML evaluation will use basic methods.")


class AdvancedDataPreprocessor:
    """
    Advanced data preprocessor using Quantum Kernel + PocketFence Kernel
    
    Features:
    - Safety filtering (PocketFence)
    - Semantic deduplication (Quantum)
    - Intelligent categorization (Quantum)
    - Quality scoring (Quantum)
    - Standardization (Quantum)
    - Dimensionality reduction (PCA/SVD) for data compression
    """
    
    def __init__(self, pocketfence_url: str = "http://localhost:5000", 
                 dedup_threshold: float = 0.9,
                 use_quantum: bool = True,
                 enable_compression: bool = True,
                 compression_ratio: float = 0.5,
                 compression_method: str = 'pca',
                 enable_scrubbing: bool = True,
                 scrubbing_options: Optional[Dict[str, bool]] = None,
                 use_advanced_scrubbing: bool = True):
        self.quantum_kernel = get_kernel(KernelConfig(use_sentence_transformers=True)) if use_quantum else None
        self.pocketfence_url = pocketfence_url
        self.dedup_threshold = dedup_threshold
        self.use_quantum = use_quantum
        self.enable_compression = enable_compression
        self.compression_ratio = compression_ratio  # 0.5 = 50% of original dimensions
        self.compression_method = compression_method  # 'pca', 'svd', 'autoencoder'
        self.pocketfence_available = self._check_pocketfence()
        
        # Data scrubbing
        self.enable_scrubbing = enable_scrubbing and SCRUBBING_AVAILABLE
        self.use_advanced_scrubbing = use_advanced_scrubbing
        self.scrubbing_options = scrubbing_options or {
            'remove_html': True,
            'remove_urls': True,
            'remove_emails': True,
            'remove_phone': True,
            'normalize_unicode': True,
            'normalize_whitespace': True,
            'remove_special_chars': False,
            'lowercase': False,
            'fix_encoding': True
        }
        
        if self.enable_scrubbing:
            if self.use_advanced_scrubbing:
                self.scrubber = AdvancedDataScrubber()
            else:
                self.scrubber = DataScrubber()
        else:
            self.scrubber = None
        
        # Compression models (fitted on data)
        self.pca_model = None
        self.svd_model = None
        self.scaler = None
        
        # Statistics
        self.stats = {
            'total_processed': 0,
            'unsafe_filtered': 0,
            'duplicates_removed': 0,
            'categories_created': 0,
            'processing_times': [],
            'compression_applied': False,
            'original_dim': 0,
            'compressed_dim': 0,
            'compression_ratio_achieved': 0.0,
            'scrubbing_applied': False,
            'noise_removed': 0,
            'low_quality_removed': 0
        }
    
    def _check_pocketfence(self) -> bool:
        """Check if PocketFence service is available"""
        try:
            response = requests.get(f"{self.pocketfence_url}/api/kernel/health", timeout=1)
            return response.status_code == 200
        except:
            return False
    
    def preprocess(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """
        Complete preprocessing pipeline
        
        Args:
            raw_data: List of raw text items to preprocess
            verbose: Print detailed progress
            
        Returns:
            Dictionary with preprocessing results
        """
        start_time = time.time()
        
        if verbose:
            print(f"[Preprocessing] Input: {len(raw_data)} items")
        
        results = {
            'original_count': len(raw_data),
            'scrubbed_data': [],
            'scrubbing_stats': {},
            'safe_data': [],
            'unsafe_data': [],
            'deduplicated': [],
            'duplicates': [],
            'categorized': {},
            'quality_scores': [],
            'final_count': 0,
            'processing_time': 0.0,
            'stats': {}
        }
        
        # Stage 0: Data scrubbing (if enabled)
        if self.enable_scrubbing and self.scrubber:
            if verbose:
                print("[Stage 0] Data Scrubbing")
            
            if self.use_advanced_scrubbing:
                # Advanced scrubbing with quality filtering
                scrubbing_results = self.scrubber.scrub_batch_advanced(
                    raw_data,
                    options=self.scrubbing_options,
                    filter_noise=True,
                    filter_low_quality=False  # Let quality scoring handle this
                )
                scrubbed_data = scrubbing_results['clean_texts']
                results['scrubbing_stats'] = {
                    'total_scrubbed': scrubbing_results['total'],
                    'kept': scrubbing_results['kept'],
                    'filtered': scrubbing_results['filtered'],
                    'noise_removed': len([f for f in scrubbing_results['filtered_out'] if f['reason'] == 'noise']),
                    'low_quality_removed': len([f for f in scrubbing_results['filtered_out'] if f['reason'] == 'low_quality']),
                    'scrubber_stats': scrubbing_results['stats']
                }
                self.stats['noise_removed'] += results['scrubbing_stats']['noise_removed']
                self.stats['low_quality_removed'] += results['scrubbing_stats']['low_quality_removed']
            else:
                # Basic scrubbing
                scrubbing_results = self.scrubber.scrub_batch(raw_data, options=self.scrubbing_options)
                scrubbed_data = [r['scrubbed'] for r in scrubbing_results]
                results['scrubbing_stats'] = {
                    'total_scrubbed': len(raw_data),
                    'scrubber_stats': self.scrubber.get_stats()
                }
            
            results['scrubbed_data'] = scrubbed_data
            self.stats['scrubbing_applied'] = True
            
            if verbose:
                print(f"  Scrubbed: {len(scrubbed_data)} items")
                if self.use_advanced_scrubbing:
                    print(f"  Noise removed: {results['scrubbing_stats']['noise_removed']}")
                    print(f"  Low quality removed: {results['scrubbing_stats']['low_quality_removed']}")
        else:
            scrubbed_data = raw_data
            results['scrubbed_data'] = scrubbed_data
        
        # Stage 1: Safety filtering (on scrubbed data)
        if verbose:
            print("[Stage 1] Safety Filtering (PocketFence Kernel)")
        safe_data, unsafe_data = self._safety_filter(scrubbed_data, verbose)
        results['safe_data'] = safe_data
        results['unsafe_data'] = unsafe_data
        self.stats['unsafe_filtered'] += len(unsafe_data)
        
        # Stage 2: Semantic deduplication
        if verbose:
            print(f"[Stage 2] Semantic Deduplication (Quantum Kernel)")
        unique_data, duplicates = self._deduplicate_semantic(safe_data, verbose)
        results['deduplicated'] = unique_data
        results['duplicates'] = duplicates
        self.stats['duplicates_removed'] += len(duplicates)
        
        # Stage 3: Categorization
        if verbose:
            print("[Stage 3] Categorization (Quantum Kernel)")
        categorized = self._categorize(unique_data, verbose)
        results['categorized'] = categorized
        self.stats['categories_created'] = len(categorized)
        
        # Stage 4: Quality scoring
        if verbose:
            print("[Stage 4] Quality Scoring (Quantum Kernel)")
        quality_scores = self._quality_score(unique_data)
        results['quality_scores'] = quality_scores
        
        # Stage 5: Dimensionality reduction (compression)
        if self.enable_compression and self.use_quantum and self.quantum_kernel:
            if verbose:
                print("[Stage 5] Dimensionality Reduction (Compression)")
            compressed_embeddings, compression_info = self._compress_embeddings(unique_data, verbose)
            results['compressed_embeddings'] = compressed_embeddings
            results['compression_info'] = compression_info
            self.stats['compression_applied'] = True
            self.stats['original_dim'] = compression_info.get('original_dim', 0)
            self.stats['compressed_dim'] = compression_info.get('compressed_dim', 0)
            self.stats['compression_ratio_achieved'] = compression_info.get('compression_ratio', 0.0)
        
        # Final results
        results['final_count'] = len(unique_data)
        results['processing_time'] = time.time() - start_time
        results['stats'] = {
            'unsafe_filtered': len(unsafe_data),
            'duplicates_removed': len(duplicates),
            'categories': len(categorized),
            'avg_quality': sum(s['score'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0,
            'compression_applied': self.stats.get('compression_applied', False),
            'compression_ratio': self.stats.get('compression_ratio_achieved', 0.0)
        }
        
        self.stats['total_processed'] += len(raw_data)
        self.stats['processing_times'].append(results['processing_time'])
        
        return results
    
    def _safety_filter(self, data: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
        """Stage 1: Filter unsafe content using PocketFence"""
        safe = []
        unsafe = []
        
        if not self.pocketfence_available:
            if verbose:
                print("  [Note] PocketFence service not available, skipping safety filter")
            return data, []
        
        for item in data:
            try:
                response = requests.post(
                    f"{self.pocketfence_url}/api/filter/content",
                    json={"content": item},
                    timeout=1
                )
                if response.status_code == 200:
                    result = response.json()
                    if result.get('isBlocked', False) or not result.get('isChildSafe', True):
                        unsafe.append(item)
                    else:
                        safe.append(item)
                else:
                    safe.append(item)
            except:
                safe.append(item)
        
        if verbose:
            print(f"  Safe: {len(safe)}, Unsafe: {len(unsafe)}")
        
        return safe, unsafe
    
    def _deduplicate_semantic(self, data: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
        """Stage 2: Remove semantic duplicates using Quantum Kernel"""
        if not self.use_quantum or not self.quantum_kernel:
            # Fallback to exact matching
            return self._deduplicate_exact(data, verbose)
        
        unique = []
        duplicates = []
        seen_embeddings = []
        
        for item in data:
            embedding = self.quantum_kernel.embed(item)
            
            # Check similarity to seen items
            is_duplicate = False
            for seen_emb in seen_embeddings:
                similarity = float(np.abs(np.dot(embedding, seen_emb)))
                if similarity >= self.dedup_threshold:
                    duplicates.append(item)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(item)
                seen_embeddings.append(embedding)
        
        if verbose:
            print(f"  Unique: {len(unique)}, Duplicates: {len(duplicates)}")
        
        return unique, duplicates
    
    def _deduplicate_exact(self, data: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
        """Fallback: Exact duplicate removal"""
        unique = []
        duplicates = []
        seen = set()
        
        for item in data:
            item_lower = item.lower().strip()
            if item_lower in seen:
                duplicates.append(item)
            else:
                unique.append(item)
                seen.add(item_lower)
        
        if verbose:
            print(f"  Unique: {len(unique)}, Duplicates: {len(duplicates)}")
        
        return unique, duplicates
    
    def _categorize(self, data: List[str], verbose: bool = False) -> Dict[str, List[str]]:
        """Stage 3: Categorize by semantic similarity"""
        if not self.use_quantum or not self.quantum_kernel:
            return self._categorize_keyword(data, verbose)
        
        categories = defaultdict(list)
        
        # Define category examples
        category_examples = {
            'technical': ['programming', 'code', 'algorithm', 'software', 'development', 'python', 'javascript'],
            'business': ['revenue', 'profit', 'market', 'sales', 'customer', 'business', 'money'],
            'support': ['help', 'issue', 'problem', 'error', 'fix', 'support', 'troubleshoot'],
            'education': ['learn', 'tutorial', 'course', 'teach', 'education', 'study'],
            'general': ['hello', 'thanks', 'information', 'question', 'general']
        }
        
        for item in data:
            best_category = 'general'
            best_score = 0.0
            
            for category, examples in category_examples.items():
                similarities = [
                    self.quantum_kernel.similarity(item, example)
                    for example in examples
                ]
                avg_similarity = sum(similarities) / len(similarities)
                
                if avg_similarity > best_score:
                    best_score = avg_similarity
                    best_category = category
            
            categories[best_category].append(item)
        
        if verbose:
            print(f"  Categories: {len(categories)}")
            for cat, items in categories.items():
                print(f"    - {cat}: {len(items)} items")
        
        return dict(categories)
    
    def _categorize_keyword(self, data: List[str], verbose: bool = False) -> Dict[str, List[str]]:
        """Fallback: Keyword-based categorization"""
        categories = defaultdict(list)
        
        category_keywords = {
            'technical': ['code', 'programming', 'algorithm', 'software', 'python', 'javascript'],
            'business': ['revenue', 'profit', 'market', 'sales', 'customer', 'business'],
            'support': ['help', 'issue', 'problem', 'error', 'fix', 'support'],
            'education': ['learn', 'tutorial', 'course', 'teach', 'education'],
            'general': []
        }
        
        for item in data:
            item_lower = item.lower()
            best_category = 'general'
            best_matches = 0
            
            for category, keywords in category_keywords.items():
                matches = sum(1 for kw in keywords if kw in item_lower)
                if matches > best_matches:
                    best_matches = matches
                    best_category = category
            
            categories[best_category].append(item)
        
        if verbose:
            print(f"  Categories: {len(categories)}")
        
        return dict(categories)
    
    def _quality_score(self, data: List[str]) -> List[Dict[str, Any]]:
        """Stage 4: Score data quality"""
        scored = []
        
        for item in data:
            length = len(item)
            word_count = len(item.split())
            
            # Length score
            if 20 <= length <= 500:
                length_score = 1.0
            elif length < 20:
                length_score = length / 20.0
            else:
                length_score = max(0.5, 1.0 - (length - 500) / 1000.0)
            
            # Completeness score
            completeness_score = min(word_count / 10.0, 1.0)
            
            # Combined quality
            quality = (length_score * 0.4 + completeness_score * 0.6)
            
            scored.append({
                'item': item,
                'score': quality,
                'length': length,
                'word_count': word_count
            })
        
        return scored
    
    def _compress_embeddings(self, data: List[str], verbose: bool = False) -> Tuple[Optional[np.ndarray], Dict[str, Any]]:
        """
        Stage 5: Compress embeddings using dimensionality reduction
        
        Args:
            data: List of text items
            verbose: Print detailed progress
            
        Returns:
            Tuple of (compressed_embeddings, compression_info)
        """
        if not self.use_quantum or not self.quantum_kernel:
            return None, {}
        
        if len(data) < 2:
            return None, {'error': 'Need at least 2 items for compression'}
        
        # Get original embeddings
        original_embeddings = np.array([self.quantum_kernel.embed(item) for item in data])
        original_dim = original_embeddings.shape[1]
        num_items = len(data)
        
        # PCA/SVD constraint: n_components <= min(n_samples, n_features)
        max_components = min(num_items, original_dim)
        target_dim = max(1, min(int(original_dim * self.compression_ratio), max_components))
        
        compression_info = {
            'original_dim': original_dim,
            'target_dim': target_dim,
            'compression_ratio': self.compression_ratio,
            'method': self.compression_method,
            'num_items': num_items,
            'max_components': max_components
        }
        
        if verbose:
            print(f"  Original dimensions: {original_dim}")
            print(f"  Target dimensions: {target_dim}")
            print(f"  Compression ratio: {self.compression_ratio:.1%}")
            print(f"  Method: {self.compression_method.upper()}")
        
        try:
            if self.compression_method == 'pca' and SKLEARN_AVAILABLE:
                compressed_embeddings, compression_info = self._compress_pca(
                    original_embeddings, target_dim, compression_info, verbose
                )
            elif self.compression_method == 'svd' and SKLEARN_AVAILABLE:
                compressed_embeddings, compression_info = self._compress_svd(
                    original_embeddings, target_dim, compression_info, verbose
                )
            else:
                # Fallback: Simple truncation (not ideal but works)
                compressed_embeddings = original_embeddings[:, :target_dim]
                compression_info['method'] = 'truncation'
                compression_info['variance_retained'] = 0.0
                if verbose:
                    print("  Using truncation (sklearn not available)")
            
            compression_info['compressed_dim'] = compressed_embeddings.shape[1]
            compression_info['compression_ratio_achieved'] = compression_info['compressed_dim'] / original_dim
            compression_info['memory_reduction'] = 1.0 - compression_info['compression_ratio_achieved']
            
            if verbose:
                print(f"  Compressed dimensions: {compression_info['compressed_dim']}")
                if 'variance_retained' in compression_info:
                    print(f"  Variance retained: {compression_info['variance_retained']:.2%}")
                print(f"  Memory reduction: {compression_info['memory_reduction']:.1%}")
            
            return compressed_embeddings, compression_info
            
        except Exception as e:
            if verbose:
                print(f"  Compression failed: {e}")
            return None, {'error': str(e)}
    
    def _compress_pca(self, embeddings: np.ndarray, target_dim: int, 
                     compression_info: Dict, verbose: bool = False) -> Tuple[np.ndarray, Dict]:
        """Compress using PCA"""
        # Standardize embeddings
        if self.scaler is None:
            self.scaler = StandardScaler()
            embeddings_scaled = self.scaler.fit_transform(embeddings)
        else:
            embeddings_scaled = self.scaler.transform(embeddings)
        
        # Apply PCA
        if self.pca_model is None:
            self.pca_model = PCA(n_components=target_dim)
            compressed = self.pca_model.fit_transform(embeddings_scaled)
        else:
            compressed = self.pca_model.transform(embeddings_scaled)
        
        # Calculate variance retained
        if hasattr(self.pca_model, 'explained_variance_ratio_'):
            variance_retained = float(np.sum(self.pca_model.explained_variance_ratio_))
            compression_info['variance_retained'] = variance_retained
        
        return compressed, compression_info
    
    def _compress_svd(self, embeddings: np.ndarray, target_dim: int,
                     compression_info: Dict, verbose: bool = False) -> Tuple[np.ndarray, Dict]:
        """Compress using Truncated SVD"""
        if self.svd_model is None:
            self.svd_model = TruncatedSVD(n_components=target_dim)
            compressed = self.svd_model.fit_transform(embeddings)
        else:
            compressed = self.svd_model.transform(embeddings)
        
        # Calculate variance retained
        if hasattr(self.svd_model, 'explained_variance_ratio_'):
            variance_retained = float(np.sum(self.svd_model.explained_variance_ratio_))
            compression_info['variance_retained'] = variance_retained
        
        return compressed, compression_info
    
    def decompress_embeddings(self, compressed_embeddings: np.ndarray) -> Optional[np.ndarray]:
        """
        Decompress embeddings back to original space (approximate)
        
        Note: This is approximate - some information is lost during compression
        """
        if not self.enable_compression:
            return None
        
        if self.compression_method == 'pca' and self.pca_model is not None:
            # Inverse transform
            decompressed_scaled = self.pca_model.inverse_transform(compressed_embeddings)
            if self.scaler is not None:
                decompressed = self.scaler.inverse_transform(decompressed_scaled)
            else:
                decompressed = decompressed_scaled
            return decompressed
        elif self.compression_method == 'svd' and self.svd_model is not None:
            # SVD inverse transform
            return self.svd_model.inverse_transform(compressed_embeddings)
        else:
            # Cannot decompress truncation
            return None
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0.0
        stats = {
            'total_processed': self.stats['total_processed'],
            'unsafe_filtered': self.stats['unsafe_filtered'],
            'duplicates_removed': self.stats['duplicates_removed'],
            'categories_created': self.stats['categories_created'],
            'avg_processing_time': avg_time
        }
        if self.stats.get('compression_applied'):
            stats['compression'] = {
                'original_dim': self.stats.get('original_dim', 0),
                'compressed_dim': self.stats.get('compressed_dim', 0),
                'compression_ratio': self.stats.get('compression_ratio_achieved', 0.0)
            }
        return stats


class ConventionalPreprocessor:
    """
    Conventional data preprocessor using standard methods
    
    Features:
    - Basic safety filtering (keyword-based)
    - Exact duplicate removal
    - Keyword-based categorization
    - Simple quality scoring
    """
    
    def __init__(self):
        self.stats = {
            'total_processed': 0,
            'unsafe_filtered': 0,
            'duplicates_removed': 0,
            'categories_created': 0,
            'processing_times': []
        }
    
    def preprocess(self, raw_data: List[str], verbose: bool = False) -> Dict[str, Any]:
        """Conventional preprocessing pipeline"""
        start_time = time.time()
        
        if verbose:
            print(f"[Preprocessing] Input: {len(raw_data)} items")
        
        results = {
            'original_count': len(raw_data),
            'safe_data': [],
            'unsafe_data': [],
            'deduplicated': [],
            'duplicates': [],
            'categorized': {},
            'quality_scores': [],
            'final_count': 0,
            'processing_time': 0.0,
            'stats': {}
        }
        
        # Stage 1: Basic safety filtering
        if verbose:
            print("[Stage 1] Basic Safety Filtering")
        safe_data, unsafe_data = self._safety_filter(raw_data, verbose)
        results['safe_data'] = safe_data
        results['unsafe_data'] = unsafe_data
        self.stats['unsafe_filtered'] += len(unsafe_data)
        
        # Stage 2: Exact duplicate removal
        if verbose:
            print("[Stage 2] Exact Duplicate Removal")
        unique_data, duplicates = self._deduplicate_exact(safe_data, verbose)
        results['deduplicated'] = unique_data
        results['duplicates'] = duplicates
        self.stats['duplicates_removed'] += len(duplicates)
        
        # Stage 3: Keyword-based categorization
        if verbose:
            print("[Stage 3] Keyword-Based Categorization")
        categorized = self._categorize_keyword(unique_data, verbose)
        results['categorized'] = categorized
        self.stats['categories_created'] = len(categorized)
        
        # Stage 4: Simple quality scoring
        if verbose:
            print("[Stage 4] Simple Quality Scoring")
        quality_scores = self._quality_score(unique_data)
        results['quality_scores'] = quality_scores
        
        # Final results
        results['final_count'] = len(unique_data)
        results['processing_time'] = time.time() - start_time
        results['stats'] = {
            'unsafe_filtered': len(unsafe_data),
            'duplicates_removed': len(duplicates),
            'categories': len(categorized),
            'avg_quality': sum(s['score'] for s in quality_scores) / len(quality_scores) if quality_scores else 0.0
        }
        
        self.stats['total_processed'] += len(raw_data)
        self.stats['processing_times'].append(results['processing_time'])
        
        return results
    
    def _safety_filter(self, data: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
        """Basic keyword-based safety filtering"""
        unsafe_keywords = ['spam', 'scam', 'hack', 'virus']  # Simplified
        safe = []
        unsafe = []
        
        for item in data:
            item_lower = item.lower()
            if any(kw in item_lower for kw in unsafe_keywords):
                unsafe.append(item)
            else:
                safe.append(item)
        
        if verbose:
            print(f"  Safe: {len(safe)}, Unsafe: {len(unsafe)}")
        
        return safe, unsafe
    
    def _deduplicate_exact(self, data: List[str], verbose: bool = False) -> Tuple[List[str], List[str]]:
        """Exact duplicate removal"""
        unique = []
        duplicates = []
        seen = set()
        
        for item in data:
            item_lower = item.lower().strip()
            if item_lower in seen:
                duplicates.append(item)
            else:
                unique.append(item)
                seen.add(item_lower)
        
        if verbose:
            print(f"  Unique: {len(unique)}, Duplicates: {len(duplicates)}")
        
        return unique, duplicates
    
    def _categorize_keyword(self, data: List[str], verbose: bool = False) -> Dict[str, List[str]]:
        """Keyword-based categorization"""
        categories = defaultdict(list)
        
        category_keywords = {
            'technical': ['code', 'programming', 'algorithm', 'software', 'python', 'javascript'],
            'business': ['revenue', 'profit', 'market', 'sales', 'customer', 'business'],
            'support': ['help', 'issue', 'problem', 'error', 'fix', 'support'],
            'education': ['learn', 'tutorial', 'course', 'teach', 'education'],
            'general': []
        }
        
        for item in data:
            item_lower = item.lower()
            best_category = 'general'
            best_matches = 0
            
            for category, keywords in category_keywords.items():
                matches = sum(1 for kw in keywords if kw in item_lower)
                if matches > best_matches:
                    best_matches = matches
                    best_category = category
            
            categories[best_category].append(item)
        
        if verbose:
            print(f"  Categories: {len(categories)}")
        
        return dict(categories)
    
    def _quality_score(self, data: List[str]) -> List[Dict[str, Any]]:
        """Simple quality scoring"""
        scored = []
        
        for item in data:
            length = len(item)
            word_count = len(item.split())
            
            # Simple scoring
            length_score = min(length / 100.0, 1.0)
            word_score = min(word_count / 10.0, 1.0)
            quality = (length_score + word_score) / 2.0
            
            scored.append({
                'item': item,
                'score': quality,
                'length': length,
                'word_count': word_count
            })
        
        return scored
    
    def get_stats(self) -> Dict[str, Any]:
        """Get preprocessing statistics"""
        avg_time = sum(self.stats['processing_times']) / len(self.stats['processing_times']) if self.stats['processing_times'] else 0.0
        return {
            'total_processed': self.stats['total_processed'],
            'unsafe_filtered': self.stats['unsafe_filtered'],
            'duplicates_removed': self.stats['duplicates_removed'],
            'categories_created': self.stats['categories_created'],
            'avg_processing_time': avg_time
        }
