"""
Data Scrubbing Tools
Enhancements for AdvancedDataPreprocessor
Provides comprehensive data cleaning and scrubbing capabilities
"""
import re
import unicodedata
from typing import List, Dict, Any, Optional, Callable
from collections import Counter
import html


class DataScrubber:
    """
    Data scrubbing tools for text cleaning and normalization
    
    Features:
    - HTML/XML tag removal
    - URL and email handling
    - Special character normalization
    - Unicode normalization
    - Whitespace cleaning
    - Encoding fixes
    - Phone number handling
    - Text normalization (lowercase, etc.)
    - Noise removal
    - Duplicate whitespace removal
    """
    
    def __init__(self):
        self.stats = {
            'html_tags_removed': 0,
            'urls_removed': 0,
            'emails_removed': 0,
            'phone_numbers_removed': 0,
            'special_chars_normalized': 0,
            'whitespace_cleaned': 0,
            'unicode_normalized': 0,
            'total_scrubbed': 0
        }
    
    def scrub(self, text: str, options: Optional[Dict[str, bool]] = None) -> Dict[str, Any]:
        """
        Comprehensive data scrubbing
        
        Args:
            text: Text to scrub
            options: Dictionary of scrubbing options:
                - remove_html: Remove HTML/XML tags (default: True)
                - remove_urls: Remove URLs (default: True)
                - remove_emails: Remove email addresses (default: True)
                - remove_phone: Remove phone numbers (default: True)
                - normalize_unicode: Normalize unicode (default: True)
                - normalize_whitespace: Normalize whitespace (default: True)
                - remove_special_chars: Remove special characters (default: False)
                - lowercase: Convert to lowercase (default: False)
                - fix_encoding: Fix encoding issues (default: True)
        
        Returns:
            Dictionary with scrubbed text and statistics
        """
        if options is None:
            options = {
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
        
        original_text = text
        scrubbed_text = text
        
        # Fix encoding issues
        if options.get('fix_encoding', True):
            scrubbed_text = self._fix_encoding(scrubbed_text)
        
        # Remove HTML/XML tags
        if options.get('remove_html', True):
            scrubbed_text, count = self._remove_html_tags(scrubbed_text)
            self.stats['html_tags_removed'] += count
        
        # Remove URLs
        if options.get('remove_urls', True):
            scrubbed_text, count = self._remove_urls(scrubbed_text)
            self.stats['urls_removed'] += count
        
        # Remove email addresses
        if options.get('remove_emails', True):
            scrubbed_text, count = self._remove_emails(scrubbed_text)
            self.stats['emails_removed'] += count
        
        # Remove phone numbers
        if options.get('remove_phone', True):
            scrubbed_text, count = self._remove_phone_numbers(scrubbed_text)
            self.stats['phone_numbers_removed'] += count
        
        # Normalize unicode
        if options.get('normalize_unicode', True):
            scrubbed_text = self._normalize_unicode(scrubbed_text)
            self.stats['unicode_normalized'] += 1
        
        # Normalize whitespace
        if options.get('normalize_whitespace', True):
            scrubbed_text = self._normalize_whitespace(scrubbed_text)
            self.stats['whitespace_cleaned'] += 1
        
        # Remove special characters
        if options.get('remove_special_chars', False):
            scrubbed_text = self._remove_special_chars(scrubbed_text)
            self.stats['special_chars_normalized'] += 1
        
        # Convert to lowercase
        if options.get('lowercase', False):
            scrubbed_text = scrubbed_text.lower()
        
        self.stats['total_scrubbed'] += 1
        
        return {
            'original': original_text,
            'scrubbed': scrubbed_text,
            'changes_made': scrubbed_text != original_text,
            'length_change': len(scrubbed_text) - len(original_text)
        }
    
    def scrub_batch(self, texts: List[str], options: Optional[Dict[str, bool]] = None) -> List[Dict[str, Any]]:
        """
        Scrub multiple texts
        
        Args:
            texts: List of texts to scrub
            options: Scrubbing options
            
        Returns:
            List of scrubbing results
        """
        return [self.scrub(text, options) for text in texts]
    
    def _fix_encoding(self, text: str) -> str:
        """Fix encoding issues"""
        try:
            # Try to decode and re-encode
            text = text.encode('utf-8', errors='ignore').decode('utf-8')
        except:
            pass
        return text
    
    def _remove_html_tags(self, text: str) -> tuple:
        """Remove HTML/XML tags"""
        # Count tags before removal
        tag_count = len(re.findall(r'<[^>]+>', text))
        
        # Remove HTML/XML tags
        text = re.sub(r'<[^>]+>', '', text)
        
        # Decode HTML entities
        text = html.unescape(text)
        
        return text, tag_count
    
    def _remove_urls(self, text: str) -> tuple:
        """Remove URLs"""
        # URL pattern
        url_pattern = r'https?://[^\s]+|www\.[^\s]+|[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}[^\s]*'
        urls = re.findall(url_pattern, text)
        text = re.sub(url_pattern, '', text)
        return text, len(urls)
    
    def _remove_emails(self, text: str) -> tuple:
        """Remove email addresses"""
        email_pattern = r'\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b'
        emails = re.findall(email_pattern, text)
        text = re.sub(email_pattern, '', text)
        return text, len(emails)
    
    def _remove_phone_numbers(self, text: str) -> tuple:
        """Remove phone numbers"""
        # Various phone number patterns
        phone_patterns = [
            r'\b\d{3}[-.]?\d{3}[-.]?\d{4}\b',  # US format
            r'\(\d{3}\)\s?\d{3}[-.]?\d{4}',     # (123) 456-7890
            r'\+\d{1,3}[-.\s]?\d{1,4}[-.\s]?\d{1,4}[-.\s]?\d{1,9}',  # International
        ]
        
        phone_count = 0
        for pattern in phone_patterns:
            phones = re.findall(pattern, text)
            phone_count += len(phones)
            text = re.sub(pattern, '', text)
        
        return text, phone_count
    
    def _normalize_unicode(self, text: str) -> str:
        """Normalize unicode characters"""
        # Normalize to NFC (Canonical Composition)
        text = unicodedata.normalize('NFC', text)
        return text
    
    def _normalize_whitespace(self, text: str) -> str:
        """Normalize whitespace"""
        # Replace multiple whitespace with single space
        text = re.sub(r'\s+', ' ', text)
        # Strip leading/trailing whitespace
        text = text.strip()
        return text
    
    def _remove_special_chars(self, text: str) -> str:
        """Remove special characters, keep alphanumeric and basic punctuation"""
        # Keep letters, numbers, spaces, and basic punctuation
        text = re.sub(r'[^a-zA-Z0-9\s.,!?;:\'"]', '', text)
        return text
    
    def get_stats(self) -> Dict[str, int]:
        """Get scrubbing statistics"""
        return self.stats.copy()
    
    def reset_stats(self):
        """Reset statistics"""
        self.stats = {
            'html_tags_removed': 0,
            'urls_removed': 0,
            'emails_removed': 0,
            'phone_numbers_removed': 0,
            'special_chars_normalized': 0,
            'whitespace_cleaned': 0,
            'unicode_normalized': 0,
            'total_scrubbed': 0
        }


class AdvancedDataScrubber(DataScrubber):
    """
    Advanced data scrubbing with additional features
    
    Additional features:
    - Noise detection and removal
    - Language detection (basic)
    - Profanity filtering (basic)
    - Spam detection (basic patterns)
    - Text quality assessment
    - Custom scrubber functions
    """
    
    def __init__(self):
        super().__init__()
        self.custom_scrubbers: List[Callable[[str], str]] = []
        self.noise_patterns = [
            r'^[^\w\s]+$',  # Only special characters
            r'^\s*$',       # Only whitespace
            r'^.{0,2}$',    # Very short (0-2 chars)
        ]
    
    def add_custom_scrubber(self, scrubber_func: Callable[[str], str]):
        """
        Add custom scrubbing function
        
        Args:
            scrubber_func: Function that takes text and returns scrubbed text
        """
        self.custom_scrubbers.append(scrubber_func)
    
    def detect_noise(self, text: str) -> bool:
        """
        Detect if text is noise
        
        Args:
            text: Text to check
            
        Returns:
            True if text appears to be noise
        """
        for pattern in self.noise_patterns:
            if re.match(pattern, text):
                return True
        return False
    
    def remove_noise(self, texts: List[str]) -> tuple:
        """
        Remove noisy texts
        
        Args:
            texts: List of texts
            
        Returns:
            Tuple of (clean_texts, noisy_texts)
        """
        clean = []
        noisy = []
        
        for text in texts:
            if self.detect_noise(text):
                noisy.append(text)
            else:
                clean.append(text)
        
        return clean, noisy
    
    def assess_quality(self, text: str) -> Dict[str, Any]:
        """
        Assess text quality
        
        Args:
            text: Text to assess
            
        Returns:
            Quality assessment dictionary
        """
        length = len(text)
        word_count = len(text.split())
        char_diversity = len(set(text.lower())) / max(len(text), 1)
        
        # Check for common quality issues
        issues = []
        if length < 10:
            issues.append('too_short')
        if word_count < 3:
            issues.append('too_few_words')
        if char_diversity < 0.1:
            issues.append('low_diversity')
        if text.isupper() and len(text) > 5:
            issues.append('all_caps')
        if re.search(r'[!]{3,}', text):
            issues.append('excessive_punctuation')
        
        quality_score = 1.0
        if 'too_short' in issues:
            quality_score -= 0.3
        if 'too_few_words' in issues:
            quality_score -= 0.2
        if 'low_diversity' in issues:
            quality_score -= 0.2
        if 'all_caps' in issues:
            quality_score -= 0.1
        if 'excessive_punctuation' in issues:
            quality_score -= 0.1
        
        quality_score = max(0.0, quality_score)
        
        return {
            'quality_score': quality_score,
            'length': length,
            'word_count': word_count,
            'char_diversity': char_diversity,
            'issues': issues,
            'is_high_quality': quality_score >= 0.7
        }
    
    def scrub_advanced(self, text: str, options: Optional[Dict[str, bool]] = None,
                      apply_custom: bool = True) -> Dict[str, Any]:
        """
        Advanced scrubbing with quality assessment
        
        Args:
            text: Text to scrub
            options: Scrubbing options
            apply_custom: Apply custom scrubbers
            
        Returns:
            Dictionary with scrubbed text, quality assessment, and statistics
        """
        # Basic scrubbing
        result = self.scrub(text, options)
        scrubbed_text = result['scrubbed']
        
        # Apply custom scrubbers
        if apply_custom:
            for scrubber in self.custom_scrubbers:
                scrubbed_text = scrubber(scrubbed_text)
        
        # Quality assessment
        quality = self.assess_quality(scrubbed_text)
        
        # Noise detection
        is_noise = self.detect_noise(scrubbed_text)
        
        result.update({
            'scrubbed': scrubbed_text,
            'quality': quality,
            'is_noise': is_noise,
            'should_keep': not is_noise and quality['is_high_quality']
        })
        
        return result
    
    def scrub_batch_advanced(self, texts: List[str], 
                            options: Optional[Dict[str, bool]] = None,
                            filter_noise: bool = True,
                            filter_low_quality: bool = False,
                            min_quality: float = 0.5) -> Dict[str, Any]:
        """
        Advanced batch scrubbing with filtering
        
        Args:
            texts: List of texts to scrub
            options: Scrubbing options
            filter_noise: Filter out noisy texts
            filter_low_quality: Filter out low-quality texts
            min_quality: Minimum quality score threshold
            
        Returns:
            Dictionary with scrubbed texts, filtered texts, and statistics
        """
        results = []
        filtered_out = []
        
        for text in texts:
            result = self.scrub_advanced(text, options)
            results.append(result)
            
            # Filter if needed
            should_filter = False
            if filter_noise and result['is_noise']:
                should_filter = True
                filtered_out.append({
                    'text': text,
                    'reason': 'noise',
                    'result': result
                })
            elif filter_low_quality and result['quality']['quality_score'] < min_quality:
                should_filter = True
                filtered_out.append({
                    'text': text,
                    'reason': 'low_quality',
                    'result': result
                })
        
        # Get clean texts
        clean_texts = [r['scrubbed'] for r in results 
                      if not (filter_noise and r['is_noise']) 
                      and not (filter_low_quality and r['quality']['quality_score'] < min_quality)]
        
        return {
            'results': results,
            'clean_texts': clean_texts,
            'filtered_out': filtered_out,
            'total': len(texts),
            'kept': len(clean_texts),
            'filtered': len(filtered_out),
            'stats': self.get_stats()
        }
