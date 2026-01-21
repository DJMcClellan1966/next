# Data Scrubbing Tools Guide

## Overview

**Data Scrubbing Tools** enhance AdvancedDataPreprocessor with comprehensive data cleaning capabilities. These tools clean, normalize, and prepare text data before it enters the preprocessing pipeline.

---

## Features

### Basic Scrubbing

- ✅ **HTML/XML Tag Removal** - Removes HTML and XML tags
- ✅ **URL Removal** - Removes URLs from text
- ✅ **Email Removal** - Removes email addresses
- ✅ **Phone Number Removal** - Removes phone numbers
- ✅ **Unicode Normalization** - Normalizes unicode characters
- ✅ **Whitespace Normalization** - Cleans whitespace
- ✅ **Encoding Fixes** - Fixes encoding issues
- ✅ **Special Character Handling** - Optional special character removal

### Normalization and Standardization

- ✅ **Number Normalization** - Converts written numbers to digits, normalizes formats
- ✅ **Date Normalization** - Converts various date formats to standard format (YYYY-MM-DD)
- ✅ **Currency Normalization** - Converts currency symbols to codes (USD, EUR, GBP)
- ✅ **Unit Normalization** - Adds spaces between numbers and units (5kg → 5 kg)
- ✅ **Punctuation Standardization** - Standardizes quotes, dashes, ellipsis
- ✅ **Abbreviation Expansion** - Expands common abbreviations (Dr. → Doctor, etc.)
- ✅ **Accent Removal** - Removes accents from characters (é → e, ñ → n)

### Advanced Scrubbing

- ✅ **Noise Detection** - Detects and removes noisy text
- ✅ **Quality Assessment** - Assesses text quality
- ✅ **Custom Scrubbers** - Add custom scrubbing functions
- ✅ **Batch Processing** - Process multiple texts efficiently
- ✅ **Filtering** - Filter noise and low-quality text

---

## Usage with AdvancedDataPreprocessor

### Basic Usage

```python
from data_preprocessor import AdvancedDataPreprocessor

# Enable scrubbing (default: True)
preprocessor = AdvancedDataPreprocessor(
    enable_scrubbing=True,  # Enable data scrubbing
    use_advanced_scrubbing=True  # Use advanced scrubbing
)

# Preprocess with scrubbing
results = preprocessor.preprocess(texts, verbose=True)

# Access scrubbed data
scrubbed_texts = results['scrubbed_data']
scrubbing_stats = results['scrubbing_stats']
```

### Custom Scrubbing Options

```python
# Customize scrubbing options
preprocessor = AdvancedDataPreprocessor(
    enable_scrubbing=True,
    scrubbing_options={
        'remove_html': True,      # Remove HTML tags
        'remove_urls': True,      # Remove URLs
        'remove_emails': True,    # Remove emails
        'remove_phone': True,     # Remove phone numbers
        'normalize_unicode': True, # Normalize unicode
        'normalize_whitespace': True, # Normalize whitespace
        'remove_special_chars': False, # Don't remove special chars
        'lowercase': False,       # Don't lowercase
        'fix_encoding': True,     # Fix encoding
        # Normalization options
        'normalize_numbers': True,    # Normalize numbers
        'normalize_dates': True,      # Normalize dates
        'normalize_currency': True,   # Normalize currency
        'normalize_units': True,      # Normalize units
        'standardize_punctuation': True, # Standardize punctuation
        'expand_abbreviations': True,   # Expand abbreviations
        'remove_accents': False        # Remove accents (optional)
    }
)

results = preprocessor.preprocess(texts)
```

### Standalone Usage

```python
from data_scrubbing import DataScrubber, AdvancedDataScrubber

# Basic scrubbing
scrubber = DataScrubber()
result = scrubber.scrub("Text with <html>tags</html> and https://example.com")
print(result['scrubbed'])  # Clean text

# Advanced scrubbing
advanced_scrubber = AdvancedDataScrubber()
result = advanced_scrubber.scrub_advanced(
    "Text with noise!!!",
    options={'remove_special_chars': True}
)
print(result['scrubbed'])  # Clean text
print(result['quality'])   # Quality assessment
print(result['is_noise'])  # Noise detection
```

---

## Scrubbing Stages

### Stage 0: Data Scrubbing (New)

**Before:** Raw data → Safety filtering → Deduplication → ...

**After:** Raw data → **Data Scrubbing** → Safety filtering → Deduplication → ...

**What it does:**
1. Removes HTML/XML tags
2. Removes URLs, emails, phone numbers
3. Normalizes unicode and whitespace
4. Fixes encoding issues
5. Detects and removes noise (advanced)
6. Assesses quality (advanced)

---

## Examples

### Example 1: Basic Scrubbing

```python
from data_scrubbing import DataScrubber

scrubber = DataScrubber()

text = "Visit https://example.com or email test@example.com or call (555) 123-4567"
result = scrubber.scrub(text)

print(f"Original: {result['original']}")
print(f"Scrubbed: {result['scrubbed']}")
print(f"Changes made: {result['changes_made']}")
```

**Output:**
```
Original: Visit https://example.com or email test@example.com or call (555) 123-4567
Scrubbed: Visit  or email  or call 
Changes made: True
```

### Example 2: Advanced Scrubbing with Quality

```python
from data_scrubbing import AdvancedDataScrubber

scrubber = AdvancedDataScrubber()

text = "This is high quality text with proper formatting."
result = scrubber.scrub_advanced(text)

print(f"Scrubbed: {result['scrubbed']}")
print(f"Quality score: {result['quality']['quality_score']:.2f}")
print(f"Is high quality: {result['quality']['is_high_quality']}")
print(f"Is noise: {result['is_noise']}")
print(f"Should keep: {result['should_keep']}")
```

**Output:**
```
Scrubbed: This is high quality text with proper formatting.
Quality score: 1.00
Is high quality: True
Is noise: False
Should keep: True
```

### Example 3: Batch Scrubbing with Filtering

```python
from data_scrubbing import AdvancedDataScrubber

scrubber = AdvancedDataScrubber()

texts = [
    "Good quality text here",
    "!!!",  # Noise
    "Visit https://example.com",
    "a",  # Too short
    "Normal text with proper content"
]

results = scrubber.scrub_batch_advanced(
    texts,
    filter_noise=True,
    filter_low_quality=True,
    min_quality=0.5
)

print(f"Total: {results['total']}")
print(f"Kept: {results['kept']}")
print(f"Filtered: {results['filtered']}")
print(f"\nClean texts:")
for text in results['clean_texts']:
    print(f"  - {text}")
```

**Output:**
```
Total: 5
Kept: 3
Filtered: 2

Clean texts:
  - Good quality text here
  - Visit 
  - Normal text with proper content
```

### Example 4: With AdvancedDataPreprocessor

```python
from data_preprocessor import AdvancedDataPreprocessor

# Create preprocessor with scrubbing
preprocessor = AdvancedDataPreprocessor(
    enable_scrubbing=True,
    use_advanced_scrubbing=True,
    scrubbing_options={
        'remove_html': True,
        'remove_urls': True,
        'remove_emails': True,
        'normalize_whitespace': True
    }
)

# Preprocess
texts = [
    "Visit <a href='https://example.com'>example</a>",
    "Email: test@example.com",
    "Normal text here"
]

results = preprocessor.preprocess(texts, verbose=True)

print(f"\nScrubbing Stats:")
print(f"  Total scrubbed: {results['scrubbing_stats']['total_scrubbed']}")
print(f"  Kept: {results['scrubbing_stats']['kept']}")
print(f"  Filtered: {results['scrubbing_stats']['filtered']}")

print(f"\nScrubbed texts:")
for text in results['scrubbed_data']:
    print(f"  - {text}")
```

---

## Scrubbing Options

### Available Options

| Option | Default | Description |
|--------|---------|-------------|
| `remove_html` | True | Remove HTML/XML tags |
| `remove_urls` | True | Remove URLs |
| `remove_emails` | True | Remove email addresses |
| `remove_phone` | True | Remove phone numbers |
| `normalize_unicode` | True | Normalize unicode characters |
| `normalize_whitespace` | True | Normalize whitespace |
| `remove_special_chars` | False | Remove special characters |
| `lowercase` | False | Convert to lowercase |
| `fix_encoding` | True | Fix encoding issues |
| `normalize_numbers` | False | Normalize number formats |
| `normalize_dates` | False | Normalize date formats |
| `normalize_currency` | False | Normalize currency formats |
| `normalize_units` | False | Normalize unit formats |
| `standardize_punctuation` | False | Standardize punctuation |
| `expand_abbreviations` | False | Expand abbreviations |
| `remove_accents` | False | Remove accents |

### Custom Scrubbing Functions

```python
from data_scrubbing import AdvancedDataScrubber

scrubber = AdvancedDataScrubber()

# Add custom scrubber
def remove_extra_punctuation(text):
    """Remove excessive punctuation"""
    import re
    return re.sub(r'[!]{3,}', '!', text)  # Replace !!! with !

scrubber.add_custom_scrubber(remove_extra_punctuation)

# Use custom scrubber
result = scrubber.scrub_advanced("This is great!!!")
print(result['scrubbed'])  # "This is great!"
```

---

## Quality Assessment

### Quality Metrics

- **Length** - Text length in characters
- **Word Count** - Number of words
- **Character Diversity** - Ratio of unique characters
- **Quality Score** - Overall quality (0.0-1.0)

### Quality Issues Detected

- `too_short` - Text is too short (< 10 chars)
- `too_few_words` - Too few words (< 3 words)
- `low_diversity` - Low character diversity
- `all_caps` - All uppercase text
- `excessive_punctuation` - Too many punctuation marks

### Example

```python
from data_scrubbing import AdvancedDataScrubber

scrubber = AdvancedDataScrubber()

texts = [
    "This is high quality text with proper formatting and content.",
    "!!!",  # Low quality
    "SHORT",  # Too short
    "Normal text here"
]

for text in texts:
    quality = scrubber.assess_quality(text)
    print(f"\nText: {text}")
    print(f"  Quality score: {quality['quality_score']:.2f}")
    print(f"  Issues: {quality['issues']}")
    print(f"  Is high quality: {quality['is_high_quality']}")
```

---

## Integration with AdvancedDataPreprocessor

### Preprocessing Pipeline with Scrubbing

```
Raw Data
  ↓
[Stage 0] Data Scrubbing (NEW)
  ├─ Remove HTML/XML tags
  ├─ Remove URLs, emails, phone numbers
  ├─ Normalize unicode and whitespace
  ├─ Fix encoding issues
  ├─ Detect noise (advanced)
  └─ Assess quality (advanced)
  ↓
[Stage 1] Safety Filtering (PocketFence)
  ↓
[Stage 2] Semantic Deduplication (Quantum)
  ↓
[Stage 3] Categorization (Quantum)
  ↓
[Stage 4] Quality Scoring (Quantum)
  ↓
[Stage 5] Compression (PCA/SVD)
  ↓
Clean, Processed Data
```

### Benefits

- ✅ **Cleaner data** - Removes noise and unwanted content
- ✅ **Better quality** - Filters low-quality text
- ✅ **Normalized** - Consistent formatting
- ✅ **Encoding fixed** - Handles encoding issues
- ✅ **Noise removed** - Filters out noise automatically

---

## Statistics

### Scrubbing Statistics

```python
from data_scrubbing import DataScrubber

scrubber = DataScrubber()

# Scrub some texts
scrubber.scrub("Text with <html>tags</html>")
scrubber.scrub("Visit https://example.com")
scrubber.scrub("Email: test@example.com")

# Get statistics
stats = scrubber.get_stats()
print(f"HTML tags removed: {stats['html_tags_removed']}")
print(f"URLs removed: {stats['urls_removed']}")
print(f"Emails removed: {stats['emails_removed']}")
print(f"Total scrubbed: {stats['total_scrubbed']}")
```

---

## Best Practices

### 1. Enable Scrubbing for Real-World Data

```python
# Always enable scrubbing for real-world data
preprocessor = AdvancedDataPreprocessor(
    enable_scrubbing=True,
    use_advanced_scrubbing=True
)
```

### 2. Customize Options Based on Data

```python
# For web-scraped data
preprocessor = AdvancedDataPreprocessor(
    enable_scrubbing=True,
    scrubbing_options={
        'remove_html': True,  # Important for web data
        'remove_urls': True,
        'remove_emails': True
    }
)

# For clean data
preprocessor = AdvancedDataPreprocessor(
    enable_scrubbing=True,
    scrubbing_options={
        'remove_html': False,  # Not needed
        'normalize_whitespace': True,  # Still useful
        'normalize_unicode': True
    }
)
```

### 3. Use Advanced Scrubbing for Quality Control

```python
# Use advanced scrubbing for quality filtering
preprocessor = AdvancedDataPreprocessor(
    enable_scrubbing=True,
    use_advanced_scrubbing=True  # Enables quality assessment
)
```

---

## Summary

### ✅ **Data Scrubbing Tools Provide:**

1. **HTML/XML Tag Removal** - Clean web-scraped data
2. **URL/Email/Phone Removal** - Remove contact information
3. **Unicode Normalization** - Handle encoding issues
4. **Whitespace Cleaning** - Normalize spacing
5. **Noise Detection** - Remove noisy text
6. **Quality Assessment** - Filter low-quality text
7. **Custom Scrubbing** - Add custom cleaning functions

### 🎯 **Integration:**

- ✅ **Integrated with AdvancedDataPreprocessor**
- ✅ **Automatic scrubbing** before preprocessing
- ✅ **Configurable options**
- ✅ **Quality filtering**
- ✅ **Statistics tracking**

### 📊 **Benefits:**

- ✅ **Cleaner data** - Removes noise and unwanted content
- ✅ **Better quality** - Filters low-quality text
- ✅ **Normalized** - Consistent formatting
- ✅ **Encoding fixed** - Handles encoding issues
- ✅ **Noise removed** - Filters out noise automatically

**Data scrubbing tools enhance AdvancedDataPreprocessor with comprehensive data cleaning capabilities!** 🚀
