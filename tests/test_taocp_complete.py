"""
Tests for TAOCP Complete Algorithms
Test missing TAOCP algorithms: sorting, string, numerical, combinatorial
"""
import sys
from pathlib import Path
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from taocp_complete_algorithms import (
        TAOCPSorting,
        TAOCPString,
        TAOCPNumerical,
        TAOCPCombinatorial,
        TAOCPComplete
    )
    TAOCP_COMPLETE_AVAILABLE = True
except ImportError:
    TAOCP_COMPLETE_AVAILABLE = False
    pytestmark = pytest.mark.skip("TAOCP complete algorithms not available")


class TestTAOCPSorting:
    """Tests for additional sorting algorithms"""
    
    def test_merge_sort(self):
        """Test merge sort"""
        arr = [3, 1, 4, 1, 5, 9, 2, 6, 5, 3, 5]
        sorted_arr = TAOCPSorting.merge_sort(arr)
        assert sorted_arr == sorted(arr)
    
    def test_radix_sort(self):
        """Test radix sort"""
        arr = [170, 45, 75, 90, 2, 802, 24, 66]
        sorted_arr = TAOCPSorting.radix_sort(arr)
        assert sorted_arr == sorted(arr)
    
    def test_counting_sort(self):
        """Test counting sort"""
        arr = [4, 2, 2, 8, 3, 3, 1]
        sorted_arr = TAOCPSorting.counting_sort(arr)
        assert sorted_arr == sorted(arr)
    
    def test_bucket_sort(self):
        """Test bucket sort"""
        arr = [0.42, 0.32, 0.33, 0.52, 0.37, 0.47, 0.51]
        sorted_arr = TAOCPSorting.bucket_sort(arr)
        assert sorted_arr == sorted(arr)


class TestTAOCPString:
    """Tests for additional string algorithms"""
    
    def test_boyer_moore(self):
        """Test Boyer-Moore"""
        text = "ABAAABCD"
        pattern = "ABC"
        matches = TAOCPString.boyer_moore(text, pattern)
        assert len(matches) > 0
        for idx in matches:
            assert text[idx:idx+len(pattern)] == pattern
    
    def test_rabin_karp(self):
        """Test Rabin-Karp"""
        text = "GEEKS FOR GEEKS"
        pattern = "GEEK"
        matches = TAOCPString.rabin_karp(text, pattern)
        assert len(matches) >= 2
    
    def test_suffix_array(self):
        """Test suffix array"""
        text = "banana"
        sa = TAOCPString.suffix_array(text)
        assert len(sa) == len(text)
        # Check sorted order
        suffixes = [text[i:] for i in sa]
        assert suffixes == sorted(suffixes)


class TestTAOCPNumerical:
    """Tests for additional numerical methods"""
    
    def test_floating_point_precision(self):
        """Test floating-point precision analysis"""
        result = TAOCPNumerical.floating_point_precision_analysis(1.0, 2.0)
        assert 'machine_epsilon' in result
        assert 'precision_digits' in result
    
    def test_chi_square_test(self):
        """Test chi-square test"""
        data = [10, 10, 10, 10, 10]  # Uniform
        result = TAOCPNumerical.chi_square_test(data)
        assert 'chi_square' in result
        assert 'is_random' in result
    
    def test_kolmogorov_smirnov_test(self):
        """Test Kolmogorov-Smirnov test"""
        data = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9]
        result = TAOCPNumerical.kolmogorov_smirnov_test(data)
        assert 'D_statistic' in result
        assert 'is_uniform' in result


class TestTAOCPCombinatorial:
    """Tests for additional combinatorial algorithms"""
    
    def test_gray_code(self):
        """Test Gray code generation"""
        codes = TAOCPCombinatorial.gray_code(3)
        assert len(codes) == 8
        # Check minimal change property
        for i in range(len(codes) - 1):
            diff = sum(a != b for a, b in zip(codes[i], codes[i+1]))
            assert diff == 1
    
    def test_integer_partitions(self):
        """Test integer partition generation"""
        partitions = TAOCPCombinatorial.integer_partitions(4)
        assert len(partitions) == 5  # 4 has 5 partitions
        for part in partitions:
            assert sum(part) == 4
    
    def test_catalan_numbers(self):
        """Test Catalan number generation"""
        catalan = TAOCPCombinatorial.catalan_numbers(5)
        assert len(catalan) == 5
        assert catalan[0] == 1
        assert catalan[1] == 2
    
    def test_bell_numbers(self):
        """Test Bell number generation"""
        bell = TAOCPCombinatorial.bell_numbers(5)
        assert len(bell) == 5
        assert bell[0] == 1
    
    def test_stirling_numbers(self):
        """Test Stirling numbers"""
        s1 = TAOCPCombinatorial.stirling_numbers_first_kind(5, 2)
        s2 = TAOCPCombinatorial.stirling_numbers_second_kind(5, 2)
        assert s1 > 0
        assert s2 > 0


class TestTAOCPComplete:
    """Test unified interface"""
    
    def test_unified_interface(self):
        """Test TAOCPComplete"""
        taocp = TAOCPComplete()
        assert taocp.sorting is not None
        assert taocp.string is not None
        assert taocp.numerical is not None
        assert taocp.combinatorial is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
