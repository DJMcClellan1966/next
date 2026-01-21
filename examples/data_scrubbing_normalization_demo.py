"""
Demonstration: Data Scrubbing with Normalization and Standardization
Shows normalization and standardization features
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from data_scrubbing import DataScrubber, AdvancedDataScrubber
from data_preprocessor import AdvancedDataPreprocessor


def demonstrate_normalization():
    """Demonstrate normalization and standardization features"""
    
    print("="*80)
    print("DATA SCRUBBING: NORMALIZATION AND STANDARDIZATION DEMONSTRATION")
    print("="*80)
    
    # Example 1: Number Normalization
    print("\n[Example 1: Number Normalization]")
    print("-" * 80)
    
    scrubber = DataScrubber()
    
    texts_with_numbers = [
        "I have twenty five items",
        "The price is one thousand dollars",
        "We need 1/2 cup of flour",
        "The score was 3,000 to 2,500"
    ]
    
    for text in texts_with_numbers:
        result = scrubber.scrub(
            text,
            options={
                'normalize_numbers': True,
                'normalize_whitespace': True
            }
        )
        print(f"  Original: {text}")
        print(f"  Normalized: {result['scrubbed']}")
        print()
    
    # Example 2: Date Normalization
    print("\n[Example 2: Date Normalization]")
    print("-" * 80)
    
    texts_with_dates = [
        "The event is on 12/25/2024",
        "Meeting scheduled for January 15, 2024",
        "Deadline: 15 March 2024"
    ]
    
    for text in texts_with_dates:
        result = scrubber.scrub(
            text,
            options={
                'normalize_dates': True,
                'normalize_whitespace': True
            }
        )
        print(f"  Original: {text}")
        print(f"  Normalized: {result['scrubbed']}")
        print()
    
    # Example 3: Currency Normalization
    print("\n[Example 3: Currency Normalization]")
    print("-" * 80)
    
    texts_with_currency = [
        "The price is $100",
        "Cost: €50",
        "Amount: £30",
        "Price in usd: 200"
    ]
    
    for text in texts_with_currency:
        result = scrubber.scrub(
            text,
            options={
                'normalize_currency': True,
                'normalize_whitespace': True
            }
        )
        print(f"  Original: {text}")
        print(f"  Normalized: {result['scrubbed']}")
        print()
    
    # Example 4: Unit Normalization
    print("\n[Example 4: Unit Normalization]")
    print("-" * 80)
    
    texts_with_units = [
        "The weight is 5kg",
        "Distance: 10cm",
        "Speed: 60mph",
        "Temperature: 25°C"
    ]
    
    for text in texts_with_units:
        result = scrubber.scrub(
            text,
            options={
                'normalize_units': True,
                'normalize_whitespace': True
            }
        )
        print(f"  Original: {text}")
        print(f"  Normalized: {result['scrubbed']}")
        print()
    
    # Example 5: Punctuation Standardization
    print("\n[Example 5: Punctuation Standardization]")
    print("-" * 80)
    
    texts_with_punctuation = [
        'He said "Hello"',
        'She said "Goodbye"',
        'Multiple!!! punctuation???',
        'Long dash — here'
    ]
    
    for text in texts_with_punctuation:
        result = scrubber.scrub(
            text,
            options={
                'standardize_punctuation': True,
                'normalize_whitespace': True
            }
        )
        print(f"  Original: {text}")
        print(f"  Standardized: {result['scrubbed']}")
        print()
    
    # Example 6: Abbreviation Expansion
    print("\n[Example 6: Abbreviation Expansion]")
    print("-" * 80)
    
    texts_with_abbreviations = [
        "See Dr. Smith, etc.",
        "Use examples, e.g., Python",
        "Compare A vs. B",
        "Prof. Johnson on Main St."
    ]
    
    for text in texts_with_abbreviations:
        result = scrubber.scrub(
            text,
            options={
                'expand_abbreviations': True,
                'normalize_whitespace': True
            }
        )
        print(f"  Original: {text}")
        print(f"  Expanded: {result['scrubbed']}")
        print()
    
    # Example 7: Accent Removal
    print("\n[Example 7: Accent Removal]")
    print("-" * 80)
    
    texts_with_accents = [
        "Café résumé naïve",
        "José and María",
        "São Paulo"
    ]
    
    for text in texts_with_accents:
        result = scrubber.scrub(
            text,
            options={
                'remove_accents': True,
                'normalize_whitespace': True
            }
        )
        print(f"  Original: {text}")
        print(f"  Without accents: {result['scrubbed']}")
        print()
    
    # Example 8: Combined Normalization
    print("\n[Example 8: Combined Normalization]")
    print("-" * 80)
    
    complex_texts = [
        "Price: $1,000 (one thousand USD) on 12/25/2024",
        "Weight: 5kg, Distance: 10cm, Speed: 60mph",
        "See Dr. Smith, etc. on Main St. — it's great!!!"
    ]
    
    for text in complex_texts:
        result = scrubber.scrub(
            text,
            options={
                'normalize_numbers': True,
                'normalize_dates': True,
                'normalize_currency': True,
                'normalize_units': True,
                'standardize_punctuation': True,
                'expand_abbreviations': True,
                'normalize_whitespace': True
            }
        )
        print(f"  Original: {text}")
        print(f"  Fully normalized: {result['scrubbed']}")
        print()
    
    # Example 9: With AdvancedDataPreprocessor
    print("\n[Example 9: With AdvancedDataPreprocessor]")
    print("-" * 80)
    
    preprocessor = AdvancedDataPreprocessor(
        enable_scrubbing=True,
        use_advanced_scrubbing=True,
        scrubbing_options={
            'normalize_numbers': True,
            'normalize_dates': True,
            'normalize_currency': True,
            'normalize_units': True,
            'standardize_punctuation': True,
            'expand_abbreviations': True,
            'normalize_whitespace': True,
            'normalize_unicode': True
        }
    )
    
    test_texts = [
        "Price: $100 on 12/25/2024",
        "Weight: 5kg, see Dr. Smith, etc.",
        "Visit https://example.com or email test@example.com"
    ]
    
    results = preprocessor.preprocess(test_texts, verbose=False)
    
    print("  Preprocessing with normalization:")
    print(f"    Original: {len(test_texts)} items")
    print(f"    Scrubbed: {len(results['scrubbed_data'])} items")
    print(f"    Final: {len(results['deduplicated'])} items")
    
    print("\n  Scrubbed texts:")
    for i, text in enumerate(results['scrubbed_data'], 1):
        print(f"    {i}. {text}")
    
    # Statistics
    if 'scrubbing_stats' in results and 'scrubber_stats' in results['scrubbing_stats']:
        stats = results['scrubbing_stats']['scrubber_stats']
        print("\n  Scrubbing Statistics:")
        print(f"    Numbers normalized: {stats.get('numbers_normalized', 0)}")
        print(f"    Dates normalized: {stats.get('dates_normalized', 0)}")
        print(f"    Currency normalized: {stats.get('currency_normalized', 0)}")
        print(f"    Units normalized: {stats.get('units_normalized', 0)}")
        print(f"    Punctuation standardized: {stats.get('punctuation_standardized', 0)}")
        print(f"    Abbreviations expanded: {stats.get('abbreviations_expanded', 0)}")
    
    print("\n" + "="*80)
    print("NORMALIZATION DEMONSTRATION COMPLETE")
    print("="*80 + "\n")
    
    print("Key Features:")
    print("  - Number normalization (written numbers to digits)")
    print("  - Date normalization (various formats to standard)")
    print("  - Currency normalization (symbols to codes)")
    print("  - Unit normalization (adds spaces)")
    print("  - Punctuation standardization (smart quotes, dashes)")
    print("  - Abbreviation expansion (etc., i.e., e.g., etc.)")
    print("  - Accent removal (é -> e, ñ -> n)")
    print("  - All integrated with AdvancedDataPreprocessor")
    print("="*80 + "\n")


if __name__ == "__main__":
    try:
        demonstrate_normalization()
    except Exception as e:
        print(f"[ERROR] Error: {e}")
        import traceback
        traceback.print_exc()
