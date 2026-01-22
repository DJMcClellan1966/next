"""
Tests for Sipser Methods
Test finite automata, regular languages, computability
"""
import sys
from pathlib import Path
import pytest

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    from sipser_methods import (
        FiniteAutomaton,
        NondeterministicFiniteAutomaton,
        StateMachine,
        RegularLanguageProcessor,
        ComputabilityAnalysis,
        SipserMethods
    )
    SIPSER_AVAILABLE = True
except ImportError:
    SIPSER_AVAILABLE = False
    pytestmark = pytest.mark.skip("Sipser methods not available")


class TestFiniteAutomaton:
    """Tests for DFA"""
    
    def test_simple_dfa(self):
        """Test simple DFA"""
        # DFA that accepts strings ending with '1'
        states = {'q0', 'q1'}
        alphabet = {'0', '1'}
        transitions = {
            ('q0', '0'): 'q0',
            ('q0', '1'): 'q1',
            ('q1', '0'): 'q0',
            ('q1', '1'): 'q1'
        }
        start = 'q0'
        accept = {'q1'}
        
        dfa = FiniteAutomaton(states, alphabet, transitions, start, accept)
        
        assert dfa.accepts('1')
        assert dfa.accepts('01')
        assert dfa.accepts('101')
        assert not dfa.accepts('0')
        assert not dfa.accepts('10')
    
    def test_process(self):
        """Test process method"""
        states = {'q0', 'q1'}
        alphabet = {'a', 'b'}
        transitions = {
            ('q0', 'a'): 'q1',
            ('q0', 'b'): 'q0',
            ('q1', 'a'): 'q1',
            ('q1', 'b'): 'q0'
        }
        start = 'q0'
        accept = {'q1'}
        
        dfa = FiniteAutomaton(states, alphabet, transitions, start, accept)
        accepted, sequence = dfa.process('ab')
        
        assert accepted
        assert len(sequence) == 3  # q0 -> q1 -> q0


class TestNondeterministicFiniteAutomaton:
    """Tests for NFA"""
    
    def test_simple_nfa(self):
        """Test simple NFA"""
        # NFA that accepts strings containing '11' or '00'
        states = {'q0', 'q1', 'q2', 'q3', 'q4'}
        alphabet = {'0', '1'}
        transitions = {
            ('q0', '0'): {'q0', 'q1'},
            ('q0', '1'): {'q0', 'q2'},
            ('q1', '0'): {'q3'},
            ('q2', '1'): {'q4'},
            ('q3', '0'): {'q3'},
            ('q3', '1'): {'q3'},
            ('q4', '0'): {'q4'},
            ('q4', '1'): {'q4'}
        }
        start = 'q0'
        accept = {'q3', 'q4'}
        
        nfa = NondeterministicFiniteAutomaton(states, alphabet, transitions, start, accept)
        
        assert nfa.accepts('00')
        assert nfa.accepts('11')
        assert nfa.accepts('001')
        assert nfa.accepts('110')
        assert not nfa.accepts('01')
        assert not nfa.accepts('10')
    
    def test_epsilon_closure(self):
        """Test epsilon closure"""
        states = {'q0', 'q1', 'q2'}
        alphabet = {'a'}
        transitions = {
            ('q0', None): {'q1'},
            ('q1', None): {'q2'},
            ('q0', 'a'): {'q1'}
        }
        start = 'q0'
        accept = {'q2'}
        
        nfa = NondeterministicFiniteAutomaton(states, alphabet, transitions, start, accept)
        closure = nfa.epsilon_closure({'q0'})
        
        assert 'q0' in closure
        assert 'q1' in closure
        assert 'q2' in closure
    
    def test_nfa_to_dfa(self):
        """Test NFA to DFA conversion"""
        # NFA that accepts strings ending with '1'
        states = {'q0', 'q1'}
        alphabet = {'0', '1'}
        transitions = {
            ('q0', '0'): {'q0'},
            ('q0', '1'): {'q0', 'q1'},
            ('q1', '0'): {'q0'},
            ('q1', '1'): {'q0', 'q1'}
        }
        start = 'q0'
        accept = {'q1'}
        
        nfa = NondeterministicFiniteAutomaton(states, alphabet, transitions, start, accept)
        dfa = nfa.to_dfa()
        
        # Test that DFA accepts same language (strings ending with '1')
        assert dfa.accepts('1')
        assert dfa.accepts('01')
        assert dfa.accepts('101')
        assert not dfa.accepts('0')
        assert not dfa.accepts('10')


class TestStateMachine:
    """Tests for state machine"""
    
    def test_state_machine(self):
        """Test state machine"""
        transitions = {
            ('idle', 'start'): 'processing',
            ('processing', 'complete'): 'done',
            ('processing', 'error'): 'error',
            ('error', 'retry'): 'processing',
            ('done', 'reset'): 'idle'
        }
        
        sm = StateMachine('idle', transitions)
        
        assert sm.get_state() == 'idle'
        assert sm.transition('start')
        assert sm.get_state() == 'processing'
        assert sm.transition('complete')
        assert sm.get_state() == 'done'
        
        history = sm.get_history()
        assert len(history) == 3


class TestRegularLanguageProcessor:
    """Tests for regular language processing"""
    
    def test_matches_pattern(self):
        """Test pattern matching"""
        assert RegularLanguageProcessor.matches_pattern(r'^\d+$', '123')
        assert not RegularLanguageProcessor.matches_pattern(r'^\d+$', 'abc')
    
    def test_find_all_matches(self):
        """Test find all matches"""
        text = 'The cat sat on the mat'
        matches = RegularLanguageProcessor.find_all_matches(r'\b\w{3}\b', text)
        assert 'The' in matches or 'cat' in matches or 'sat' in matches
    
    def test_extract_groups(self):
        """Test extract groups"""
        text = 'Date: 2024-01-20'
        groups = RegularLanguageProcessor.extract_groups(r'(\d{4})-(\d{2})-(\d{2})', text)
        assert len(groups) > 0
        assert len(groups[0]) == 3
    
    def test_validate_format(self):
        """Test validate format"""
        assert RegularLanguageProcessor.validate_format(r'^\d{3}-\d{2}-\d{4}$', '123-45-6789')
        assert not RegularLanguageProcessor.validate_format(r'^\d{3}-\d{2}-\d{4}$', '123456789')


class TestComputabilityAnalysis:
    """Tests for computability analysis"""
    
    def test_is_decidable(self):
        """Test decidability checking"""
        # Undecidable
        result = ComputabilityAnalysis.is_decidable('halting problem')
        assert result is False
        
        # Decidable
        result = ComputabilityAnalysis.is_decidable('regular language recognition')
        assert result is True
        
        # Unknown
        result = ComputabilityAnalysis.is_decidable('some random problem')
        assert result is None
    
    def test_classify_problem(self):
        """Test problem classification"""
        classification = ComputabilityAnalysis.classify_problem('classify images')
        assert classification['type'] == 'ml_problem'
        assert classification['category'] == 'machine_learning'
        
        classification = ComputabilityAnalysis.classify_problem('recognize regular language')
        assert classification['type'] == 'language_recognition'
    
    def test_reduce_problem(self):
        """Test problem reduction"""
        result = ComputabilityAnalysis.reduce_problem('pattern matching', 'string matching')
        assert result  # Should have common keywords


class TestSipserMethods:
    """Test unified framework"""
    
    def test_unified_interface(self):
        """Test SipserMethods"""
        sipser = SipserMethods()
        
        assert sipser.finite_automaton is not None
        assert sipser.nfa is not None
        assert sipser.state_machine is not None
        assert sipser.regular_language is not None
        assert sipser.computability is not None


if __name__ == '__main__':
    pytest.main([__file__, '-v', '--tb=short'])
