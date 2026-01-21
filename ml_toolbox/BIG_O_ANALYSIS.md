# Big O Notation Analysis for ML Toolbox

## Overview

This document provides Big O complexity analysis for all ML Toolbox operations, helping you understand performance characteristics and optimize your code.

---

## Compartment 1: Data

### AdvancedDataPreprocessor

#### `preprocess(data: List[str])` - O(n²) worst case, O(n log n) average

**Breakdown:**
- **Stage 0: Data Scrubbing** - O(n) where n is number of items
- **Stage 1: Safety Filtering** - O(n) - linear scan
- **Stage 2: Semantic Deduplication** - O(n²) worst case, O(n log n) with optimizations
  - Embedding creation: O(n * d) where d is embedding dimension
  - Similarity comparison: O(n²) for all pairs
  - With caching: O(n log n) average
- **Stage 3: Categorization** - O(n * k) where k is number of categories
- **Stage 4: Quality Scoring** - O(n)
- **Stage 5: Compression** - O(n * d²) for PCA/SVD

**Optimizations:**
- ✅ Caching reduces semantic deduplication to O(n log n)
- ✅ Batch processing: O(n/p) where p is parallel workers
- ✅ Early termination in similarity search

**Total: O(n²) worst case, O(n log n) with optimizations**

#### `_deduplicate_semantic(data)` - O(n²) worst case, O(n log n) with caching

**Without caching:**
- Create embeddings: O(n * d)
- Compare all pairs: O(n²)
- **Total: O(n²)**

**With caching:**
- Create embeddings: O(n * d) with O(1) cache lookups
- Compare pairs: O(n log n) with early termination
- **Total: O(n log n)**

#### `_compress_embeddings(data)` - O(n * d²)

- Embedding creation: O(n * d)
- PCA/SVD computation: O(n * d²)
- **Total: O(n * d²)**

**Memory: O(n * d)**

---

## Compartment 2: Infrastructure

### Quantum Kernel

#### `embed(text: str)` - O(1) with cache, O(d) without

- **With cache:** O(1) - hash lookup
- **Without cache:** O(d) where d is embedding dimension
- **Memory: O(d)**

#### `similarity(text1, text2)` - O(1) with cache, O(d) without

- **With cache:** O(1) - hash lookup
- **Without cache:** O(d) - dot product
- **Memory: O(1)**

#### `find_similar(query, candidates, top_k)` - O(n log k)

- Embed query: O(1) with cache
- Compare to candidates: O(n * d)
- Sort and select top_k: O(n log k)
- **Total: O(n log k)** where n is candidates, k is top_k

#### `build_relationship_graph(texts)` - O(n²)

- Compare all pairs: O(n²)
- Build graph: O(n²)
- **Total: O(n²)**

**Optimization:** Incremental updates - O(n) for adding one item

---

## Compartment 3: Algorithms

### ML Evaluation

#### `evaluate_model(model, X, y, cv=5)` - O(cv * n * f(n))

- Cross-validation: O(cv) folds
- Model training per fold: O(n * f(n)) where f(n) is model complexity
- **Total: O(cv * n * f(n))**

**Model complexities:**
- Linear models: O(n * d)
- Tree models: O(n * d * log n)
- Neural networks: O(n * d * e) where e is epochs

### Hyperparameter Tuning

#### `tune(model, X, y, param_grid, method='grid')` - O(p * cv * n * f(n))

- **Grid search:** O(p * cv * n * f(n)) where p is parameter combinations
- **Random search:** O(i * cv * n * f(n)) where i is iterations
- **Total: O(p * cv * n * f(n))**

### Ensemble Learning

#### `create_voting_ensemble(base_models, X, y)` - O(m * n * f(n))

- Train m models: O(m * n * f(n))
- Voting: O(m * n)
- **Total: O(m * n * f(n))**

#### `create_bagging_ensemble(base_model, X, y, n_estimators)` - O(e * n * f(n))

- Train e estimators: O(e * n * f(n))
- Aggregation: O(e * n)
- **Total: O(e * n * f(n))**

---

## Optimizations

### Caching

#### LRU Cache Operations

- `get(key)`: **O(1)** - hash lookup + OrderedDict move
- `put(key, value)`: **O(1)** - hash insert + OrderedDict insert
- `evict()`: **O(1)** - OrderedDict popitem

**Memory: O(c)** where c is cache size

### Parallel Processing

#### Batch Processing

- **Sequential:** O(n * f(n))
- **Parallel:** O(n/p * f(n)) where p is workers
- **Speedup:** ~p times faster (with overhead)

### Memory Optimization

#### Array Optimization

- `optimize_array(array)`: **O(n)** - linear scan
- **Memory savings:** 50% for float64→float32, 50% for int64→int32

---

## Performance Guidelines

### For Small Datasets (< 1,000 items)

- ✅ Use standard preprocessing
- ✅ No need for batch processing
- ✅ Caching provides minimal benefit
- **Expected time:** < 1 second

### For Medium Datasets (1,000 - 10,000 items)

- ✅ Use caching
- ✅ Consider batch processing
- ✅ Use cross-validation (5-fold)
- **Expected time:** 1-10 seconds

### For Large Datasets (10,000 - 100,000 items)

- ✅ Use batch processing
- ✅ Enable parallel processing
- ✅ Use compression
- ✅ Aggressive caching
- **Expected time:** 10-100 seconds

### For Very Large Datasets (> 100,000 items)

- ✅ Use Advanced Compartment 1 (Big Data)
- ✅ Batch processing with parallel workers
- ✅ Memory-efficient operations
- ✅ Streaming processing
- **Expected time:** Minutes to hours

---

## Optimization Strategies

### 1. Use Caching

```python
# Enable caching (default: True)
results = toolbox.data.preprocess(texts, use_cache=True)
```

**Benefit:** O(n²) → O(n log n) for repeated operations

### 2. Use Batch Processing

```python
# For large datasets
results = advanced_toolbox.big_data.process_in_batches(
    large_texts,
    batch_size=1000,
    parallel=True
)
```

**Benefit:** O(n²) → O(n/p) with p workers

### 3. Use Compression

```python
# Reduce embedding dimensions
preprocessor = AdvancedDataPreprocessor(
    enable_compression=True,
    compression_ratio=0.5  # 50% reduction
)
```

**Benefit:** O(n * d²) → O(n * (d/2)²) = 4x faster

### 4. Optimize Memory

```python
from ml_toolbox.optimizations import MemoryOptimizer

optimizer = MemoryOptimizer()
X_optimized = optimizer.optimize_array(X)
```

**Benefit:** 50% memory reduction

---

## Summary Table

| Operation | Worst Case | Average (Optimized) | Memory |
|-----------|------------|---------------------|--------|
| `preprocess()` | O(n²) | O(n log n) | O(n * d) |
| `embed()` | O(d) | O(1) cached | O(d) |
| `similarity()` | O(d) | O(1) cached | O(1) |
| `find_similar()` | O(n * d) | O(n log k) | O(k) |
| `evaluate_model()` | O(cv * n * f(n)) | O(cv * n * f(n)) | O(n * d) |
| `tune()` | O(p * cv * n * f(n)) | O(i * cv * n * f(n)) | O(n * d) |
| `process_in_batches()` | O(n²) | O(n/p * f(n)) | O(batch_size * d) |

**Legend:**
- n = number of items
- d = embedding dimension
- k = top_k results
- p = parallel workers
- cv = cross-validation folds
- f(n) = model training complexity

---

## Best Practices for Performance

1. **Enable caching** for repeated operations
2. **Use batch processing** for large datasets
3. **Enable parallel processing** when possible
4. **Use compression** to reduce dimensions
5. **Monitor performance** with PerformanceMonitor
6. **Optimize memory** for large arrays
7. **Choose appropriate algorithms** based on data size

---

## Performance Monitoring

```python
from ml_toolbox.optimizations import get_global_monitor

monitor = get_global_monitor()
stats = monitor.get_all_stats()

for func_name, func_stats in stats.items():
    print(f"{func_name}:")
    print(f"  Calls: {func_stats['calls']}")
    print(f"  Avg Time: {func_stats['avg_time']:.4f}s")
    print(f"  Total Time: {func_stats['total_time']:.4f}s")
```

---

## Conclusion

The ML Toolbox is optimized for performance with:
- ✅ O(1) caching for repeated operations
- ✅ O(n/p) parallel processing
- ✅ O(n log n) average complexity
- ✅ Memory-efficient operations
- ✅ Performance monitoring

**Use optimizations for best performance!** 🚀
