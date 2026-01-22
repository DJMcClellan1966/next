"""
Sipser Methods - Introduction to the Theory of Computation
Finite automata, regular languages, and computability for ML

Methods from:
- Michael Sipser "Introduction to the Theory of Computation"
- Finite automata (DFA/NFA)
- Regular languages and pattern matching
- Computability analysis
"""
import sys
from pathlib import Path
from typing import List, Dict, Tuple, Any, Optional, Callable, Set
from collections import defaultdict, deque
import re

sys.path.insert(0, str(Path(__file__).parent))


class FiniteAutomaton:
    """
    Deterministic Finite Automaton (DFA) - Sipser
    
    Pattern matching and language recognition
    """
    
    def __init__(self, states: Set[str], alphabet: Set[str], 
                 transitions: Dict[Tuple[str, str], str],
                 start_state: str, accept_states: Set[str]):
        """
        Args:
            states: Set of states
            alphabet: Input alphabet
            transitions: Transition function (state, symbol) -> next_state
            start_state: Initial state
            accept_states: Accepting states
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
    
    def process(self, input_string: str) -> Tuple[bool, List[str]]:
        """
        Process input string
        
        Args:
            input_string: Input to process
            
        Returns:
            Tuple (accepted, state_sequence)
        """
        current_state = self.start_state
        state_sequence = [current_state]
        
        for symbol in input_string:
            if symbol not in self.alphabet:
                return False, state_sequence
            
            transition_key = (current_state, symbol)
            if transition_key not in self.transitions:
                return False, state_sequence
            
            current_state = self.transitions[transition_key]
            state_sequence.append(current_state)
        
        accepted = current_state in self.accept_states
        return accepted, state_sequence
    
    def accepts(self, input_string: str) -> bool:
        """Check if automaton accepts input"""
        accepted, _ = self.process(input_string)
        return accepted


class NondeterministicFiniteAutomaton:
    """
    Nondeterministic Finite Automaton (NFA) - Sipser
    
    More flexible pattern matching
    """
    
    def __init__(self, states: Set[str], alphabet: Set[str],
                 transitions: Dict[Tuple[str, Optional[str]], Set[str]],
                 start_state: str, accept_states: Set[str]):
        """
        Args:
            states: Set of states
            alphabet: Input alphabet (None for epsilon)
            transitions: Transition function (state, symbol) -> set of states
            start_state: Initial state
            accept_states: Accepting states
        """
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states
    
    def epsilon_closure(self, states: Set[str]) -> Set[str]:
        """Compute epsilon closure"""
        closure = set(states)
        queue = deque(states)
        
        while queue:
            state = queue.popleft()
            epsilon_key = (state, None)
            if epsilon_key in self.transitions:
                for next_state in self.transitions[epsilon_key]:
                    if next_state not in closure:
                        closure.add(next_state)
                        queue.append(next_state)
        
        return closure
    
    def process(self, input_string: str) -> bool:
        """
        Process input string
        
        Args:
            input_string: Input to process
            
        Returns:
            True if accepted
        """
        current_states = self.epsilon_closure({self.start_state})
        
        for symbol in input_string:
            if symbol not in self.alphabet:
                return False
            
            next_states = set()
            for state in current_states:
                transition_key = (state, symbol)
                if transition_key in self.transitions:
                    next_states.update(self.transitions[transition_key])
            
            current_states = self.epsilon_closure(next_states)
        
        return bool(current_states & self.accept_states)
    
    def accepts(self, input_string: str) -> bool:
        """Check if NFA accepts input"""
        return self.process(input_string)
    
    def to_dfa(self) -> FiniteAutomaton:
        """
        Convert NFA to DFA (subset construction)
        
        Returns:
            Equivalent DFA
        """
        dfa_states = set()
        dfa_transitions = {}
        dfa_start = tuple(sorted(self.epsilon_closure({self.start_state})))
        dfa_states.add(dfa_start)
        dfa_accept = set()
        
        queue = deque([dfa_start])
        
        while queue:
            current_dfa_state = queue.popleft()
            current_nfa_states = set(current_dfa_state)
            
            # Check if accepting
            if current_nfa_states & self.accept_states:
                dfa_accept.add(current_dfa_state)
            
            # Process each symbol
            for symbol in self.alphabet:
                next_nfa_states = set()
                for nfa_state in current_nfa_states:
                    transition_key = (nfa_state, symbol)
                    if transition_key in self.transitions:
                        next_nfa_states.update(self.transitions[transition_key])
                
                if next_nfa_states:
                    next_nfa_states = self.epsilon_closure(next_nfa_states)
                    next_dfa_state = tuple(sorted(next_nfa_states))
                    
                    if next_dfa_state not in dfa_states:
                        dfa_states.add(next_dfa_state)
                        queue.append(next_dfa_state)
                    
                    dfa_transitions[(current_dfa_state, symbol)] = next_dfa_state
        
        # Convert tuple states to string states
        state_map = {state: f"q{i}" for i, state in enumerate(dfa_states)}
        dfa_states_str = set(state_map.values())
        dfa_transitions_str = {
            (state_map[s], sym): state_map[t]
            for (s, sym), t in dfa_transitions.items()
        }
        dfa_start_str = state_map[dfa_start]
        dfa_accept_str = {state_map[s] for s in dfa_accept}
        
        return FiniteAutomaton(
            dfa_states_str,
            self.alphabet,
            dfa_transitions_str,
            dfa_start_str,
            dfa_accept_str
        )


class StateMachine:
    """
    General State Machine Framework
    
    For ML workflow and process modeling
    """
    
    def __init__(self, initial_state: str, transitions: Dict[Tuple[str, str], str]):
        """
        Args:
            initial_state: Initial state
            transitions: (current_state, event) -> next_state
        """
        self.current_state = initial_state
        self.transitions = transitions
        self.state_history = [initial_state]
    
    def transition(self, event: str) -> bool:
        """
        Transition on event
        
        Args:
            event: Event to process
            
        Returns:
            True if transition successful
        """
        key = (self.current_state, event)
        if key in self.transitions:
            self.current_state = self.transitions[key]
            self.state_history.append(self.current_state)
            return True
        return False
    
    def get_state(self) -> str:
        """Get current state"""
        return self.current_state
    
    def get_history(self) -> List[str]:
        """Get state history"""
        return self.state_history.copy()


class RegularLanguageProcessor:
    """
    Regular Language Processing - Sipser
    
    Pattern matching and language recognition
    """
    
    @staticmethod
    def matches_pattern(pattern: str, text: str) -> bool:
        """
        Match regular expression pattern
        
        Args:
            pattern: Regular expression pattern
            text: Text to match
            
        Returns:
            True if pattern matches
        """
        try:
            return bool(re.match(pattern, text))
        except re.error:
            return False
    
    @staticmethod
    def find_all_matches(pattern: str, text: str) -> List[str]:
        """
        Find all matches of pattern in text
        
        Args:
            pattern: Regular expression pattern
            text: Text to search
            
        Returns:
            List of matches
        """
        try:
            return re.findall(pattern, text)
        except re.error:
            return []
    
    @staticmethod
    def extract_groups(pattern: str, text: str) -> List[Tuple]:
        """
        Extract groups from pattern
        
        Args:
            pattern: Regular expression with groups
            text: Text to process
            
        Returns:
            List of group tuples
        """
        try:
            matches = re.finditer(pattern, text)
            return [match.groups() for match in matches]
        except re.error:
            return []
    
    @staticmethod
    def validate_format(pattern: str, text: str) -> bool:
        """
        Validate text format against pattern
        
        Args:
            pattern: Validation pattern
            text: Text to validate
            
        Returns:
            True if valid
        """
        try:
            return bool(re.fullmatch(pattern, text))
        except re.error:
            return False


class ComputabilityAnalysis:
    """
    Computability Analysis - Sipser
    
    Decidability and problem classification
    """
    
    @staticmethod
    def is_decidable(problem_description: str) -> Optional[bool]:
        """
        Determine if problem is decidable (simplified)
        
        Args:
            problem_description: Problem description
            
        Returns:
            True if decidable, False if undecidable, None if unknown
        """
        # Known undecidable problems
        undecidable_keywords = [
            'halting problem',
            'turing machine acceptance',
            'post correspondence',
            'rice theorem'
        ]
        
        problem_lower = problem_description.lower()
        for keyword in undecidable_keywords:
            if keyword in problem_lower:
                return False
        
        # Known decidable problems
        decidable_keywords = [
            'regular language',
            'context-free language',
            'finite automaton',
            'regular expression'
        ]
        
        for keyword in decidable_keywords:
            if keyword in problem_lower:
                return True
        
        return None  # Unknown
    
    @staticmethod
    def classify_problem(problem_description: str) -> Dict[str, Any]:
        """
        Classify problem by type
        
        Args:
            problem_description: Problem description
            
        Returns:
            Classification dictionary
        """
        problem_lower = problem_description.lower()
        
        classification = {
            'type': 'unknown',
            'decidable': None,
            'complexity': 'unknown',
            'category': 'unknown'
        }
        
        # Language recognition
        if any(kw in problem_lower for kw in ['language', 'pattern', 'string']):
            classification['type'] = 'language_recognition'
            classification['category'] = 'automata'
        
        # Decision problem
        if any(kw in problem_lower for kw in ['decide', 'determine', 'check']):
            classification['type'] = 'decision_problem'
        
        # Optimization problem
        if any(kw in problem_lower for kw in ['optimize', 'minimize', 'maximize']):
            classification['type'] = 'optimization_problem'
            classification['category'] = 'optimization'
        
        # ML problem
        if any(kw in problem_lower for kw in ['classify', 'predict', 'learn', 'train']):
            classification['type'] = 'ml_problem'
            classification['category'] = 'machine_learning'
        
        # Decidability
        classification['decidable'] = ComputabilityAnalysis.is_decidable(problem_description)
        
        return classification
    
    @staticmethod
    def reduce_problem(problem_a: str, problem_b: str) -> bool:
        """
        Check if problem A reduces to problem B (simplified)
        
        Args:
            problem_a: Problem A
            problem_b: Problem B
            
        Returns:
            True if A reduces to B
        """
        # Simplified: check if problems are related
        a_lower = problem_a.lower()
        b_lower = problem_b.lower()
        
        # Check for common keywords
        a_keywords = set(a_lower.split())
        b_keywords = set(b_lower.split())
        
        common = a_keywords & b_keywords
        return len(common) > 0


class SipserMethods:
    """
    Unified Sipser Methods Framework
    """
    
    def __init__(self):
        self.finite_automaton = FiniteAutomaton
        self.nfa = NondeterministicFiniteAutomaton
        self.state_machine = StateMachine
        self.regular_language = RegularLanguageProcessor()
        self.computability = ComputabilityAnalysis()
    
    def get_dependencies(self) -> Dict[str, str]:
        """Get dependencies"""
        return {
            'python': 'Python 3.8+',
            're': 'Python re module (built-in)'
        }
