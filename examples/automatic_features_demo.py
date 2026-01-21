"""
Demonstration: Automatic Feature Creation
Shows what features AdvancedDataPreprocessor creates automatically
"""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_preprocessor import AdvancedDataPreprocessor


def demonstrate_automatic_features():
    """Demonstrate all automatic features created by AdvancedDataPreprocessor"""
    
    print("="*80)
    print("AUTOMATIC FEATURE CREATION DEMONSTRATION")
    print("="*80)
    
    # Sample texts
    texts = [
        "Python programming is great for data science and machine learning",
        "Machine learning uses neural networks to recognize patterns",
        "Revenue increased by twenty percent this quarter",
        "Customer satisfaction drives business growth and profitability",
        "I need help with technical issues in my code",
        "Support team provides assistance for troubleshooting problems",
        "Learn Python programming through online courses and tutorials",
        "Educational content helps students understand complex concepts"
    ]
    
    print(f"\n[Input Data]")
    print(f"  Number of texts: {len(texts)}")
    for i, text in enumerate(texts, 1):
        print(f"  {i}. {text[:60]}...")
    
    # Preprocess
    print(f"\n[Preprocessing with AdvancedDataPreprocessor]")
    print("-" * 80)
    
    preprocessor = AdvancedDataPreprocessor(
        dedup_threshold=0.75,  # Lower threshold to keep more samples
        enable_compression=True,
        compression_ratio=0.5
    )
    
    results = preprocessor.preprocess(texts, verbose=True)
    
    # Feature 1: Semantic Embeddings
    print("\n" + "="*80)
    print("FEATURE 1: SEMANTIC EMBEDDINGS")
    print("="*80)
    
    embeddings = []
    for text in results['deduplicated']:
        embedding = preprocessor.quantum_kernel.embed(text)
        embeddings.append(embedding)
    
    embeddings_array = np.array(embeddings)
    print(f"\nEmbedding Shape: {embeddings_array.shape}")
    print(f"  - Samples: {embeddings_array.shape[0]}")
    print(f"  - Dimensions: {embeddings_array.shape[1]}")
    print(f"\nSample Embedding (first 10 values):")
    print(f"  {embeddings_array[0][:10]}")
    print(f"\nUse Case: Direct input to neural networks")
    
    # Feature 2: Compressed Embeddings
    print("\n" + "="*80)
    print("FEATURE 2: COMPRESSED EMBEDDINGS")
    print("="*80)
    
    if 'compressed_embeddings' in results and results['compressed_embeddings'] is not None:
        compressed = results['compressed_embeddings']
        compression_info = results['compression_info']
        
        print(f"\nCompressed Embedding Shape: {compressed.shape}")
        print(f"  - Original dimensions: {compression_info.get('original_dim', 'N/A')}")
        print(f"  - Compressed dimensions: {compressed.shape[1]}")
        print(f"  - Compression ratio: {compression_info.get('compression_ratio', 0):.1%}")
        print(f"  - Variance retained: {compression_info.get('variance_retained', 0):.1%}")
        print(f"  - Memory reduction: {compression_info.get('memory_reduction', 0):.1%}")
        print(f"\nSample Compressed Embedding (first 10 values):")
        print(f"  {compressed[0][:10]}")
        print(f"\nUse Case: Faster training, lower memory usage")
    else:
        print("\nCompression not applied (enable_compression=False or insufficient data)")
    
    # Feature 3: Category Labels
    print("\n" + "="*80)
    print("FEATURE 3: CATEGORY LABELS")
    print("="*80)
    
    categories = results['categorized']
    print(f"\nCategories Created: {len(categories)}")
    
    category_labels = []
    for text in results['deduplicated']:
        found = False
        for cat, items in categories.items():
            if text in items:
                category_labels.append(cat)
                found = True
                break
        if not found:
            category_labels.append('general')
    
    print(f"\nCategory Distribution:")
    from collections import Counter
    category_counts = Counter(category_labels)
    for cat, count in category_counts.items():
        print(f"  - {cat}: {count} items")
    
    # One-hot encode categories
    df_categories = pd.get_dummies(category_labels)
    print(f"\nOne-Hot Encoded Categories Shape: {df_categories.shape}")
    print(f"  - Samples: {df_categories.shape[0]}")
    print(f"  - Categories: {df_categories.shape[1]}")
    print(f"\nCategory Features:")
    print(df_categories.head())
    print(f"\nUse Case: Add category as feature column")
    
    # Feature 4: Quality Scores
    print("\n" + "="*80)
    print("FEATURE 4: QUALITY SCORES")
    print("="*80)
    
    quality_scores = results['quality_scores']
    print(f"\nQuality Scores Created: {len(quality_scores)}")
    
    print(f"\nQuality Metrics:")
    for i, score_data in enumerate(quality_scores[:5], 1):  # Show first 5
        print(f"\n  {i}. Text: {score_data['item'][:50]}...")
        print(f"     Quality Score: {score_data['score']:.3f}")
        print(f"     Length: {score_data['length']} characters")
        print(f"     Word Count: {score_data['word_count']} words")
    
    # Extract quality features
    quality_features = np.array([
        [s['score'], s['length'], s['word_count']]
        for s in quality_scores
    ])
    
    print(f"\nQuality Features Shape: {quality_features.shape}")
    print(f"  - Samples: {quality_features.shape[0]}")
    print(f"  - Features: {quality_features.shape[1]} (score, length, word_count)")
    print(f"\nUse Case: Filter low-quality data, add as features")
    
    # Feature 5: Relationship Features
    print("\n" + "="*80)
    print("FEATURE 5: RELATIONSHIP FEATURES")
    print("="*80)
    
    kernel = preprocessor.quantum_kernel
    similarity_features = []
    
    print(f"\nComputing similarity features...")
    for i, text in enumerate(results['deduplicated']):
        similarities = []
        for j, other_text in enumerate(results['deduplicated']):
            if i != j:
                sim = kernel.similarity(text, other_text)
                similarities.append(sim)
        
        avg_sim = np.mean(similarities) if similarities else 0.0
        max_sim = np.max(similarities) if similarities else 0.0
        min_sim = np.min(similarities) if similarities else 0.0
        similarity_features.append([avg_sim, max_sim, min_sim])
    
    similarity_array = np.array(similarity_features)
    
    print(f"\nSimilarity Features Shape: {similarity_array.shape}")
    print(f"  - Samples: {similarity_array.shape[0]}")
    print(f"  - Features: {similarity_array.shape[1]} (avg, max, min similarity)")
    print(f"\nSample Similarity Features:")
    for i, (text, sim_features) in enumerate(zip(results['deduplicated'][:3], similarity_array[:3]), 1):
        print(f"\n  {i}. Text: {text[:50]}...")
        print(f"     Avg Similarity: {sim_features[0]:.3f}")
        print(f"     Max Similarity: {sim_features[1]:.3f}")
        print(f"     Min Similarity: {sim_features[2]:.3f}")
    print(f"\nUse Case: Find similar items, graph neural networks")
    
    # Combine All Features
    print("\n" + "="*80)
    print("COMBINED FEATURE SET")
    print("="*80)
    
    # Combine all features
    feature_list = []
    feature_names = []
    
    # 1. Compressed embeddings (or original if not compressed)
    if 'compressed_embeddings' in results and results['compressed_embeddings'] is not None:
        feature_list.append(results['compressed_embeddings'])
        feature_names.append(f"Compressed Embeddings ({results['compressed_embeddings'].shape[1]} dims)")
    else:
        feature_list.append(embeddings_array)
        feature_names.append(f"Original Embeddings ({embeddings_array.shape[1]} dims)")
    
    # 2. Category features
    feature_list.append(df_categories.values)
    feature_names.append(f"Categories ({df_categories.shape[1]} dims)")
    
    # 3. Quality features
    feature_list.append(quality_features)
    feature_names.append(f"Quality ({quality_features.shape[1]} dims)")
    
    # 4. Similarity features
    feature_list.append(similarity_array)
    feature_names.append(f"Similarity ({similarity_array.shape[1]} dims)")
    
    # Combine
    X_combined = np.column_stack(feature_list)
    
    print(f"\nCombined Feature Matrix:")
    print(f"  Shape: {X_combined.shape}")
    print(f"  - Samples: {X_combined.shape[0]}")
    print(f"  - Total Features: {X_combined.shape[1]}")
    
    print(f"\nFeature Breakdown:")
    current_idx = 0
    for name, features in zip(feature_names, feature_list):
        n_features = features.shape[1]
        print(f"  - {name}: {n_features} features (indices {current_idx}-{current_idx+n_features-1})")
        current_idx += n_features
    
    print(f"\nUse Case: Ready for machine learning models!")
    print(f"  - Neural networks: X_combined as input")
    print(f"  - Classification: X_combined, y as labels")
    print(f"  - Clustering: X_combined for grouping")
    
    # Summary
    print("\n" + "="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nAutomatic Features Created:")
    print(f"  1. Semantic Embeddings: {embeddings_array.shape[1]} dimensions")
    if 'compressed_embeddings' in results and results['compressed_embeddings'] is not None:
        print(f"  2. Compressed Embeddings: {results['compressed_embeddings'].shape[1]} dimensions")
    print(f"  3. Category Labels: {df_categories.shape[1]} categories")
    print(f"  4. Quality Scores: {quality_features.shape[1]} metrics")
    print(f"  5. Similarity Features: {similarity_array.shape[1]} metrics")
    print(f"\nTotal Features: {X_combined.shape[1]} dimensions")
    print(f"\nAll features created automatically - no manual engineering required!")
    print("="*80 + "\n")
    
    return {
        'embeddings': embeddings_array,
        'compressed': results.get('compressed_embeddings'),
        'categories': df_categories,
        'quality': quality_features,
        'similarity': similarity_array,
        'combined': X_combined
    }


if __name__ == "__main__":
    try:
        features = demonstrate_automatic_features()
        print("\n[+] Demonstration complete!")
        print("\nKey Takeaways:")
        print("  - AdvancedDataPreprocessor creates 5 types of features automatically")
        print("  - No manual feature engineering required")
        print("  - Features are ready for machine learning models")
        print("  - Semantic understanding captures meaning, not just keywords")
    except Exception as e:
        print(f"\n[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
