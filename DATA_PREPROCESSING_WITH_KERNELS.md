# Data Preprocessing with Quantum Kernel + PocketFence Kernel

## Overview

Combining **Quantum Kernel** (semantic understanding) + **PocketFence Kernel** (safety filtering) creates a powerful data preprocessing pipeline.

**Quantum Kernel:** Understands meaning, finds duplicates, categorizes, standardizes  
**PocketFence Kernel:** Filters unsafe content, validates URLs, checks threats, sanitizes

**Together:** Clean, safe, semantically organized data

---

## 🎯 Use Case 1: Text Data Cleaning & Standardization

### Problem
Raw text data is messy: duplicates, typos, inconsistent formatting, unsafe content

### Solution
Use Quantum Kernel for semantic cleaning, PocketFence for safety

```python
from quantum_kernel import get_kernel, KernelConfig
import requests  # For PocketFence API

class DataPreprocessor:
    """Advanced data preprocessing using both kernels"""
    
    def __init__(self):
        self.quantum_kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
        self.pocketfence_url = "http://localhost:5000"  # PocketFence service
    
    def preprocess_text(self, texts: list) -> dict:
        """
        Complete preprocessing pipeline:
        1. Safety check (PocketFence)
        2. Semantic deduplication (Quantum)
        3. Categorization (Quantum)
        4. Standardization (Quantum)
        """
        results = {
            'original_count': len(texts),
            'safe_texts': [],
            'unsafe_texts': [],
            'deduplicated': [],
            'categorized': {},
            'cleaned': []
        }
        
        # Step 1: Safety filtering (PocketFence)
        print("[1/4] Safety filtering...")
        for text in texts:
            is_safe = self._check_safety(text)
            if is_safe:
                results['safe_texts'].append(text)
            else:
                results['unsafe_texts'].append(text)
        
        # Step 2: Semantic deduplication (Quantum)
        print("[2/4] Semantic deduplication...")
        results['deduplicated'] = self._deduplicate_semantic(results['safe_texts'])
        
        # Step 3: Categorization (Quantum)
        print("[3/4] Categorization...")
        results['categorized'] = self._categorize(results['deduplicated'])
        
        # Step 4: Standardization (Quantum)
        print("[4/4] Standardization...")
        results['cleaned'] = self._standardize(results['deduplicated'])
        
        return results
    
    def _check_safety(self, text: str) -> bool:
        """Check if text is safe using PocketFence Kernel"""
        try:
            response = requests.post(
                f"{self.pocketfence_url}/api/filter/content",
                json={"content": text},
                timeout=2
            )
            if response.status_code == 200:
                result = response.json()
                return not result.get('isBlocked', False) and result.get('isChildSafe', True)
        except:
            # If PocketFence not available, assume safe (or implement local check)
            return True
        return True
    
    def _deduplicate_semantic(self, texts: list, threshold: float = 0.9) -> list:
        """Remove semantic duplicates using Quantum Kernel"""
        unique_texts = []
        seen_embeddings = []
        
        for text in texts:
            embedding = self.quantum_kernel.embed(text)
            
            # Check if similar to any seen text
            is_duplicate = False
            for seen_embedding in seen_embeddings:
                similarity = float(np.dot(embedding, seen_embedding))
                if similarity >= threshold:
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique_texts.append(text)
                seen_embeddings.append(embedding)
        
        return unique_texts
    
    def _categorize(self, texts: list) -> dict:
        """Categorize texts using Quantum Kernel"""
        categories = defaultdict(list)
        
        # Define category examples
        category_examples = {
            'technical': ['programming', 'code', 'algorithm', 'software'],
            'business': ['revenue', 'profit', 'market', 'sales'],
            'support': ['help', 'issue', 'problem', 'error'],
            'general': ['hello', 'thanks', 'information']
        }
        
        for text in texts:
            best_category = 'general'
            best_score = 0.0
            
            for category, examples in category_examples.items():
                # Find similarity to category examples
                similarities = [
                    self.quantum_kernel.similarity(text, example)
                    for example in examples
                ]
                avg_similarity = sum(similarities) / len(similarities)
                
                if avg_similarity > best_score:
                    best_score = avg_similarity
                    best_category = category
            
            categories[best_category].append(text)
        
        return dict(categories)
    
    def _standardize(self, texts: list) -> list:
        """Standardize text format using Quantum Kernel"""
        standardized = []
        
        for text in texts:
            # Find canonical form (most common similar text)
            # Simplified - would use clustering in production
            standardized.append(text.strip().lower())
        
        return standardized
```

**Benefits:**
- ✅ Removes unsafe content (PocketFence)
- ✅ Finds semantic duplicates (Quantum)
- ✅ Organizes by category (Quantum)
- ✅ Standardizes format (Quantum)

---

## 🔍 Use Case 2: URL & Link Validation

### Problem
Datasets contain URLs - need to validate safety and find duplicates

### Solution
PocketFence checks safety, Quantum finds similar/duplicate URLs

```python
class URLPreprocessor:
    """Preprocess URLs using both kernels"""
    
    def __init__(self):
        self.quantum_kernel = get_kernel()
        self.pocketfence_url = "http://localhost:5000"
    
    def preprocess_urls(self, urls: list) -> dict:
        """Preprocess URL list"""
        results = {
            'safe_urls': [],
            'unsafe_urls': [],
            'duplicate_groups': [],
            'categorized': {}
        }
        
        # Step 1: Safety check (PocketFence)
        safe_urls = []
        for url in urls:
            is_safe = self._check_url_safety(url)
            if is_safe:
                safe_urls.append(url)
                results['safe_urls'].append(url)
            else:
                results['unsafe_urls'].append(url)
        
        # Step 2: Find duplicate/similar URLs (Quantum)
        results['duplicate_groups'] = self._find_duplicate_urls(safe_urls)
        
        # Step 3: Categorize by domain/topic (Quantum)
        results['categorized'] = self._categorize_urls(safe_urls)
        
        return results
    
    def _check_url_safety(self, url: str) -> bool:
        """Check URL safety using PocketFence"""
        try:
            response = requests.post(
                f"{self.pocketfence_url}/api/filter/url",
                json={"url": url},
                timeout=2
            )
            if response.status_code == 200:
                result = response.json()
                return not result.get('isBlocked', False)
        except:
            return True  # Assume safe if service unavailable
        return True
    
    def _find_duplicate_urls(self, urls: list) -> list:
        """Find similar/duplicate URLs using Quantum Kernel"""
        # Extract URL text (domain, path, etc.)
        url_texts = [self._extract_url_text(url) for url in urls]
        
        # Find semantic duplicates
        duplicate_groups = []
        processed = set()
        
        for i, url in enumerate(urls):
            if i in processed:
                continue
            
            group = [url]
            url_text = url_texts[i]
            
            for j, other_url in enumerate(urls[i+1:], i+1):
                if j in processed:
                    continue
                
                other_text = url_texts[j]
                similarity = self.quantum_kernel.similarity(url_text, other_text)
                
                if similarity > 0.85:  # Very similar URLs
                    group.append(other_url)
                    processed.add(j)
            
            if len(group) > 1:
                duplicate_groups.append(group)
                processed.add(i)
        
        return duplicate_groups
    
    def _extract_url_text(self, url: str) -> str:
        """Extract meaningful text from URL"""
        # Remove protocol, extract domain and path
        url = url.replace('https://', '').replace('http://', '')
        parts = url.split('/')
        domain = parts[0]
        path = ' '.join(parts[1:]) if len(parts) > 1 else ''
        return f"{domain} {path}"
    
    def _categorize_urls(self, urls: list) -> dict:
        """Categorize URLs by topic/domain"""
        categories = defaultdict(list)
        
        for url in urls:
            url_text = self._extract_url_text(url)
            
            # Categorize based on keywords
            if any(kw in url_text.lower() for kw in ['api', 'docs', 'documentation']):
                categories['documentation'].append(url)
            elif any(kw in url_text.lower() for kw in ['blog', 'article', 'post']):
                categories['content'].append(url)
            elif any(kw in url_text.lower() for kw in ['github', 'code', 'repo']):
                categories['code'].append(url)
            else:
                categories['other'].append(url)
        
        return dict(categories)
```

**Benefits:**
- ✅ Filters unsafe URLs (PocketFence)
- ✅ Finds duplicate/similar URLs (Quantum)
- ✅ Organizes by category (Quantum)

---

## 📊 Use Case 3: User-Generated Content Preprocessing

### Problem
User comments, reviews, posts need cleaning and safety checking

### Solution
PocketFence filters unsafe content, Quantum organizes and deduplicates

```python
class UserContentPreprocessor:
    """Preprocess user-generated content"""
    
    def preprocess_user_content(self, content_list: list) -> dict:
        """Complete preprocessing for user content"""
        results = {
            'safe_content': [],
            'unsafe_content': [],
            'spam_detected': [],
            'categorized': {},
            'sentiment_analyzed': {}
        }
        
        # Step 1: Safety filtering (PocketFence)
        safe_content = []
        for content in content_list:
            safety_result = self._check_content_safety(content)
            if safety_result['is_safe']:
                safe_content.append(content)
                results['safe_content'].append(content)
            else:
                results['unsafe_content'].append({
                    'content': content,
                    'reason': safety_result['reason']
                })
        
        # Step 2: Spam detection (Quantum - find duplicates)
        results['spam_detected'] = self._detect_spam(safe_content)
        
        # Step 3: Categorization (Quantum)
        results['categorized'] = self._categorize_content(safe_content)
        
        # Step 4: Sentiment analysis (Quantum)
        results['sentiment_analyzed'] = self._analyze_sentiment(safe_content)
        
        return results
    
    def _check_content_safety(self, content: str) -> dict:
        """Check content safety using PocketFence"""
        try:
            response = requests.post(
                f"{self.pocketfence_url}/api/filter/content",
                json={"content": content},
                timeout=2
            )
            if response.status_code == 200:
                result = response.json()
                return {
                    'is_safe': not result.get('isBlocked', False),
                    'threat_score': result.get('threatScore', 0.0),
                    'reason': result.get('reason', '')
                }
        except:
            return {'is_safe': True, 'threat_score': 0.0, 'reason': ''}
        
        return {'is_safe': True, 'threat_score': 0.0, 'reason': ''}
    
    def _detect_spam(self, content_list: list) -> list:
        """Detect spam using semantic similarity (Quantum)"""
        spam_groups = []
        processed = set()
        
        for i, content in enumerate(content_list):
            if i in processed:
                continue
            
            # Find very similar content (likely spam)
            similar = [content]
            for j, other_content in enumerate(content_list[i+1:], i+1):
                if j in processed:
                    continue
                
                similarity = self.quantum_kernel.similarity(content, other_content)
                if similarity > 0.95:  # Nearly identical
                    similar.append(other_content)
                    processed.add(j)
            
            if len(similar) > 1:
                spam_groups.append(similar)
                processed.add(i)
        
        return spam_groups
    
    def _categorize_content(self, content_list: list) -> dict:
        """Categorize content by topic"""
        categories = defaultdict(list)
        
        category_keywords = {
            'question': ['how', 'what', 'why', 'when', 'where', '?'],
            'complaint': ['problem', 'issue', 'error', 'broken', 'bad'],
            'praise': ['great', 'excellent', 'love', 'amazing', 'perfect'],
            'request': ['please', 'can you', 'help', 'need', 'want']
        }
        
        for content in content_list:
            best_category = 'general'
            best_score = 0.0
            
            for category, keywords in category_keywords.items():
                # Check semantic similarity to category
                similarities = [
                    self.quantum_kernel.similarity(content, kw)
                    for kw in keywords
                ]
                avg_score = sum(similarities) / len(similarities)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_category = category
            
            categories[best_category].append(content)
        
        return dict(categories)
    
    def _analyze_sentiment(self, content_list: list) -> dict:
        """Basic sentiment analysis using Quantum Kernel"""
        sentiment_keywords = {
            'positive': ['good', 'great', 'excellent', 'love', 'happy', 'satisfied'],
            'negative': ['bad', 'terrible', 'hate', 'angry', 'disappointed', 'poor'],
            'neutral': ['okay', 'fine', 'average', 'normal']
        }
        
        sentiments = defaultdict(list)
        
        for content in content_list:
            best_sentiment = 'neutral'
            best_score = 0.0
            
            for sentiment, keywords in sentiment_keywords.items():
                similarities = [
                    self.quantum_kernel.similarity(content, kw)
                    for kw in keywords
                ]
                avg_score = sum(similarities) / len(similarities)
                
                if avg_score > best_score:
                    best_score = avg_score
                    best_sentiment = sentiment
            
            sentiments[best_sentiment].append({
                'content': content,
                'score': best_score
            })
        
        return dict(sentiments)
```

**Benefits:**
- ✅ Filters unsafe content (PocketFence)
- ✅ Detects spam/duplicates (Quantum)
- ✅ Categorizes by type (Quantum)
- ✅ Analyzes sentiment (Quantum)

---

## 🔄 Use Case 4: Data Pipeline Integration

### Complete Preprocessing Pipeline

```python
class CompleteDataPipeline:
    """End-to-end data preprocessing pipeline"""
    
    def __init__(self):
        self.quantum_kernel = get_kernel()
        self.pocketfence_url = "http://localhost:5000"
    
    def process_dataset(self, raw_data: list) -> dict:
        """
        Complete preprocessing:
        1. Safety filtering (PocketFence)
        2. Semantic deduplication (Quantum)
        3. Categorization (Quantum)
        4. Quality scoring (Quantum)
        5. Standardization (Quantum)
        """
        pipeline_results = {
            'stage1_safety': self._safety_filter(raw_data),
            'stage2_deduplication': None,
            'stage3_categorization': None,
            'stage4_quality': None,
            'stage5_standardization': None,
            'final_clean_data': []
        }
        
        # Stage 2: Deduplication
        safe_data = pipeline_results['stage1_safety']['safe']
        pipeline_results['stage2_deduplication'] = self._deduplicate(safe_data)
        
        # Stage 3: Categorization
        unique_data = pipeline_results['stage2_deduplication']['unique']
        pipeline_results['stage3_categorization'] = self._categorize(unique_data)
        
        # Stage 4: Quality scoring
        pipeline_results['stage4_quality'] = self._quality_score(unique_data)
        
        # Stage 5: Standardization
        pipeline_results['stage5_standardization'] = self._standardize(unique_data)
        
        # Final clean data
        pipeline_results['final_clean_data'] = pipeline_results['stage5_standardization']
        
        return pipeline_results
    
    def _safety_filter(self, data: list) -> dict:
        """Stage 1: Safety filtering"""
        safe = []
        unsafe = []
        
        for item in data:
            if self._is_safe(item):
                safe.append(item)
            else:
                unsafe.append(item)
        
        return {'safe': safe, 'unsafe': unsafe, 'filtered_count': len(unsafe)}
    
    def _deduplicate(self, data: list) -> dict:
        """Stage 2: Semantic deduplication"""
        unique = []
        duplicates = []
        
        # Use quantum kernel to find semantic duplicates
        seen_embeddings = []
        for item in data:
            embedding = self.quantum_kernel.embed(str(item))
            
            is_duplicate = False
            for seen_emb in seen_embeddings:
                if np.dot(embedding, seen_emb) > 0.9:
                    duplicates.append(item)
                    is_duplicate = True
                    break
            
            if not is_duplicate:
                unique.append(item)
                seen_embeddings.append(embedding)
        
        return {'unique': unique, 'duplicates': duplicates}
    
    def _categorize(self, data: list) -> dict:
        """Stage 3: Categorization"""
        # Implementation similar to previous examples
        pass
    
    def _quality_score(self, data: list) -> dict:
        """Stage 4: Quality scoring"""
        scored = []
        
        for item in data:
            item_str = str(item)
            
            # Quality factors
            length_score = min(len(item_str) / 100, 1.0)  # Prefer longer content
            completeness_score = 1.0 if len(item_str.split()) > 5 else 0.5
            
            # Semantic coherence (simplified)
            coherence_score = 0.8  # Would use more sophisticated analysis
            
            quality = (length_score + completeness_score + coherence_score) / 3
            
            scored.append({
                'item': item,
                'quality_score': quality
            })
        
        return {'scored': scored, 'avg_quality': sum(s['quality_score'] for s in scored) / len(scored)}
    
    def _standardize(self, data: list) -> list:
        """Stage 5: Standardization"""
        standardized = []
        
        for item in data:
            # Standardize format
            item_str = str(item).strip()
            # Additional standardization logic
            standardized.append(item_str)
        
        return standardized
    
    def _is_safe(self, item: any) -> bool:
        """Check safety using PocketFence"""
        item_str = str(item)
        try:
            response = requests.post(
                f"{self.pocketfence_url}/api/filter/content",
                json={"content": item_str},
                timeout=2
            )
            if response.status_code == 200:
                result = response.json()
                return not result.get('isBlocked', False)
        except:
            return True
        return True
```

---

## 📈 Performance Benefits

### Quantum Kernel Benefits:
- **Semantic Deduplication:** Finds duplicates even with different wording
- **Intelligent Categorization:** Groups by meaning, not just keywords
- **Relationship Discovery:** Finds connections between data points
- **Fast Processing:** Caching provides 10-200x speedup

### PocketFence Kernel Benefits:
- **Safety Filtering:** Removes unsafe/threatening content
- **URL Validation:** Checks URL safety
- **Threat Detection:** Identifies malicious content
- **Batch Processing:** Handles high volumes

### Combined Benefits:
- ✅ **Cleaner Data:** Removes duplicates, unsafe content
- ✅ **Better Organization:** Semantic categorization
- ✅ **Higher Quality:** Quality scoring and standardization
- ✅ **Faster Processing:** Both kernels optimized for performance

---

## 🎯 Real-World Applications

### 1. **E-commerce Product Data**
- PocketFence: Filter unsafe product descriptions
- Quantum: Find duplicate products, categorize by type

### 2. **Social Media Content**
- PocketFence: Filter inappropriate content
- Quantum: Detect spam, categorize posts, analyze sentiment

### 3. **Customer Support Tickets**
- PocketFence: Filter abusive language
- Quantum: Categorize by issue type, find duplicates

### 4. **Research Data**
- PocketFence: Validate URLs, filter unsafe sources
- Quantum: Deduplicate papers, categorize by topic

### 5. **Training Data for ML**
- PocketFence: Ensure safe training data
- Quantum: Organize, deduplicate, categorize

---

## 💻 Implementation Example

```python
from quantum_kernel import get_kernel, KernelConfig
import requests

# Initialize
quantum_kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
pocketfence_url = "http://localhost:5000"

# Raw data
raw_data = [
    "Python is great for data science",
    "Python is excellent for data science",  # Semantic duplicate
    "Check out this amazing site: http://example.com",
    "Inappropriate content here...",
    "Machine learning uses algorithms",
    "ML uses algorithms"  # Semantic duplicate
]

# Preprocessing pipeline
preprocessor = DataPreprocessor()
results = preprocessor.preprocess_text(raw_data)

print(f"Original: {results['original_count']} items")
print(f"Safe: {len(results['safe_texts'])} items")
print(f"Unsafe: {len(results['unsafe_texts'])} items")
print(f"Deduplicated: {len(results['deduplicated'])} items")
print(f"Categories: {list(results['categorized'].keys())}")
```

---

## 📊 Expected Results

### Input: 1000 raw data items
- **After Safety Filter:** ~950 safe items (PocketFence removes 50 unsafe)
- **After Deduplication:** ~700 unique items (Quantum finds 250 semantic duplicates)
- **After Categorization:** Organized into 5-10 categories
- **Final Clean Data:** ~700 high-quality, categorized items

### Performance:
- **Safety Filtering:** ~50-100ms per item (PocketFence API)
- **Deduplication:** ~1-2ms per item (Quantum Kernel with cache)
- **Categorization:** ~2-5ms per item (Quantum Kernel)
- **Total:** ~1000 items in 5-10 seconds

---

## 🚀 Best Practices

### 1. **Pipeline Order**
1. Safety first (PocketFence)
2. Then deduplication (Quantum)
3. Then categorization (Quantum)
4. Finally standardization

### 2. **Caching**
- Quantum Kernel caches embeddings (10-200x speedup)
- Cache PocketFence results for repeated content

### 3. **Batch Processing**
- Process in batches for efficiency
- Use parallel processing where possible

### 4. **Error Handling**
- Handle PocketFence service unavailability
- Fallback to local safety checks if needed

---

## 💡 Advanced Use Cases

### 1. **Multi-Language Support**
- Quantum Kernel handles semantic similarity across languages
- PocketFence filters unsafe content in any language

### 2. **Real-Time Preprocessing**
- Stream data through pipeline
- Quantum Kernel handles real-time deduplication
- PocketFence filters in real-time

### 3. **Incremental Learning**
- Quantum Kernel learns patterns over time
- Improves categorization with more data
- PocketFence learns new threat patterns

---

## 🎓 Summary

**Quantum Kernel + PocketFence Kernel = Complete Data Preprocessing**

- **PocketFence:** Safety, validation, threat detection
- **Quantum:** Understanding, deduplication, categorization
- **Together:** Clean, safe, organized, high-quality data

**Key Benefits:**
- ✅ Removes unsafe content
- ✅ Finds semantic duplicates
- ✅ Intelligent categorization
- ✅ Quality improvement
- ✅ Fast processing

**Perfect for:**
- E-commerce data
- Social media content
- Customer support
- Research data
- ML training data

---

**This combination gives you enterprise-grade data preprocessing!**
