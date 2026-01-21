"""
Data Preprocessing Pipeline
Combines Quantum Kernel + PocketFence Kernel for complete data cleaning
"""
import sys
from pathlib import Path
import time
from collections import defaultdict

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_kernel import get_kernel, KernelConfig
import requests
import numpy as np


class DataPreprocessingPipeline:
    """Complete data preprocessing using both kernels"""
    
    def __init__(self, pocketfence_url: str = "http://localhost:5000"):
        self.quantum_kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
        self.pocketfence_url = pocketfence_url
        self.pocketfence_available = self._check_pocketfence()
    
    def _check_pocketfence(self) -> bool:
        """Check if PocketFence service is available"""
        try:
            response = requests.get(f"{self.pocketfence_url}/api/kernel/health", timeout=2)
            return response.status_code == 200
        except:
            return False
    
    def preprocess(self, raw_data: list) -> dict:
        """
        Complete preprocessing pipeline
        
        Stages:
        1. Safety filtering (PocketFence)
        2. Semantic deduplication (Quantum)
        3. Categorization (Quantum)
        4. Quality scoring (Quantum)
        """
        print("="*70)
        print("DATA PREPROCESSING PIPELINE")
        print("="*70)
        print(f"\nInput: {len(raw_data)} items")
        
        results = {
            'original_count': len(raw_data),
            'stage1_safety': {},
            'stage2_deduplication': {},
            'stage3_categorization': {},
            'stage4_quality': {},
            'final_count': 0,
            'processing_time': 0.0
        }
        
        start_time = time.time()
        
        # Stage 1: Safety filtering
        print("\n[Stage 1] Safety Filtering (PocketFence Kernel)")
        print("-" * 70)
        safe_data, unsafe_data = self._safety_filter(raw_data)
        results['stage1_safety'] = {
            'safe': safe_data,
            'unsafe': unsafe_data,
            'safe_count': len(safe_data),
            'unsafe_count': len(unsafe_data)
        }
        print(f"  Safe: {len(safe_data)} items")
        print(f"  Unsafe: {len(unsafe_data)} items")
        
        # Stage 2: Semantic deduplication
        print("\n[Stage 2] Semantic Deduplication (Quantum Kernel)")
        print("-" * 70)
        unique_data, duplicates = self._deduplicate_semantic(safe_data)
        results['stage2_deduplication'] = {
            'unique': unique_data,
            'duplicates': duplicates,
            'unique_count': len(unique_data),
            'duplicate_count': len(duplicates)
        }
        print(f"  Unique: {len(unique_data)} items")
        print(f"  Duplicates removed: {len(duplicates)} items")
        
        # Stage 3: Categorization
        print("\n[Stage 3] Categorization (Quantum Kernel)")
        print("-" * 70)
        categorized = self._categorize(unique_data)
        results['stage3_categorization'] = categorized
        print(f"  Categories: {len(categorized)}")
        for category, items in categorized.items():
            print(f"    - {category}: {len(items)} items")
        
        # Stage 4: Quality scoring
        print("\n[Stage 4] Quality Scoring (Quantum Kernel)")
        print("-" * 70)
        quality_scores = self._quality_score(unique_data)
        results['stage4_quality'] = quality_scores
        avg_quality = sum(s['score'] for s in quality_scores) / len(quality_scores) if quality_scores else 0
        print(f"  Average quality: {avg_quality:.2f}")
        print(f"  High quality (>0.7): {sum(1 for s in quality_scores if s['score'] > 0.7)} items")
        
        # Final results
        results['final_count'] = len(unique_data)
        results['processing_time'] = time.time() - start_time
        
        print("\n" + "="*70)
        print("PREPROCESSING COMPLETE")
        print("="*70)
        print(f"\n  Original: {results['original_count']} items")
        print(f"  After safety filter: {results['stage1_safety']['safe_count']} items")
        print(f"  After deduplication: {results['final_count']} items")
        print(f"  Reduction: {results['original_count'] - results['final_count']} items removed")
        print(f"  Processing time: {results['processing_time']:.2f}s")
        print("="*70 + "\n")
        
        return results
    
    def _safety_filter(self, data: list) -> tuple:
        """Stage 1: Filter unsafe content"""
        safe = []
        unsafe = []
        
        if not self.pocketfence_available:
            print("  [Note] PocketFence service not available, skipping safety filter")
            return data, []
        
        for item in data:
            item_str = str(item)
            try:
                response = requests.post(
                    f"{self.pocketfence_url}/api/filter/content",
                    json={"content": item_str},
                    timeout=1
                )
                if response.status_code == 200:
                    result = response.json()
                    if result.get('isBlocked', False) or not result.get('isChildSafe', True):
                        unsafe.append(item)
                    else:
                        safe.append(item)
                else:
                    safe.append(item)  # Assume safe if check fails
            except:
                safe.append(item)  # Assume safe if service unavailable
        
        return safe, unsafe
    
    def _deduplicate_semantic(self, data: list, threshold: float = 0.9) -> tuple:
        """Stage 2: Remove semantic duplicates"""
        unique = []
        duplicates = []
        seen_embeddings = []
        
        for item in data:
            item_str = str(item)
            embedding = self.quantum_kernel.embed(item_str)
            
            # Check similarity to seen items
            is_duplicate = False
            for seen_emb in seen_embeddings:
                similarity = float(np.abs(np.dot(embedding, seen_emb)))
                if similarity >= threshold:
                    duplicates.append(item)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(item)
                seen_embeddings.append(embedding)
        
        return unique, duplicates
    
    def _categorize(self, data: list) -> dict:
        """Stage 3: Categorize by semantic similarity"""
        categories = defaultdict(list)
        
        # Define category examples
        category_examples = {
            'technical': ['programming', 'code', 'algorithm', 'software', 'development'],
            'business': ['revenue', 'profit', 'market', 'sales', 'customer'],
            'support': ['help', 'issue', 'problem', 'error', 'fix'],
            'general': ['hello', 'thanks', 'information', 'question']
        }
        
        for item in data:
            item_str = str(item)
            best_category = 'general'
            best_score = 0.0
            
            for category, examples in category_examples.items():
                # Compute average similarity to category examples
                similarities = [
                    self.quantum_kernel.similarity(item_str, example)
                    for example in examples
                ]
                avg_similarity = sum(similarities) / len(similarities)
                
                if avg_similarity > best_score:
                    best_score = avg_similarity
                    best_category = category
            
            categories[best_category].append(item)
        
        return dict(categories)
    
    def _quality_score(self, data: list) -> list:
        """Stage 4: Score data quality"""
        scored = []
        
        for item in data:
            item_str = str(item)
            
            # Quality factors
            length = len(item_str)
            word_count = len(item_str.split())
            
            # Length score (prefer reasonable length)
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


def demo_preprocessing():
    """Demonstrate data preprocessing"""
    # Sample raw data (messy, with duplicates, unsafe content)
    raw_data = [
        "Python is great for data science",
        "Python is excellent for data science",  # Semantic duplicate
        "I love programming in Python",  # Similar to above
        "Machine learning uses algorithms",
        "ML uses algorithms",  # Semantic duplicate
        "Check out this site: http://example.com",
        "How do I fix errors in my code?",
        "I need help with programming",
        "Programming is fun and rewarding",
        "Code errors are frustrating",
        "Short",  # Low quality
        "This is a very long sentence that goes on and on and might be considered low quality due to excessive length and verbosity",  # Too long
    ]
    
    # Create pipeline
    pipeline = DataPreprocessingPipeline()
    
    # Process data
    results = pipeline.preprocess(raw_data)
    
    # Show some examples
    print("\n[Example Results]")
    print("-" * 70)
    
    if results['stage2_deduplication']['duplicates']:
        print("\nDuplicates Found:")
        for dup in results['stage2_deduplication']['duplicates'][:3]:
            print(f"  - {str(dup)[:60]}...")
    
    if results['stage3_categorization']:
        print("\nCategorized Items:")
        for category, items in list(results['stage3_categorization'].items())[:3]:
            print(f"\n  {category.upper()}:")
            for item in items[:2]:
                print(f"    - {str(item)[:60]}...")
    
    if results['stage4_quality']:
        print("\nQuality Scores (Top 3):")
        top_quality = sorted(results['stage4_quality'], key=lambda x: x['score'], reverse=True)[:3]
        for item_data in top_quality:
            print(f"  [{item_data['score']:.2f}] {str(item_data['item'])[:60]}...")


if __name__ == "__main__":
    try:
        demo_preprocessing()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
        print("\nNote: PocketFence service is optional.")
        print("      If not running, safety filtering will be skipped.")
