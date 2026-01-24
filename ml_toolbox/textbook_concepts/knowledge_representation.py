"""
Knowledge Representation (AIMA - Russell & Norvig)

Implements:
- Knowledge Base
- Rule-Based Systems
- Expert Systems
"""
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
import logging

logger = logging.getLogger(__name__)


@dataclass
class Rule:
    """Production rule"""
    condition: str  # Logical condition
    action: str  # Action/conclusion
    confidence: float = 1.0


class KnowledgeBase:
    """
    Knowledge Base
    
    Stores facts and rules for reasoning
    """
    
    def __init__(self):
        self.facts: Dict[str, Any] = {}
        self.rules: List[Rule] = []
    
    def add_fact(self, fact: str, value: Any):
        """Add fact to knowledge base"""
        self.facts[fact] = value
        logger.debug(f"[KnowledgeBase] Added fact: {fact} = {value}")
    
    def add_rule(self, condition: str, action: str, confidence: float = 1.0):
        """Add rule to knowledge base"""
        rule = Rule(condition=condition, action=action, confidence=confidence)
        self.rules.append(rule)
        logger.debug(f"[KnowledgeBase] Added rule: {condition} -> {action}")
    
    def query(self, query: str) -> Optional[Any]:
        """Query knowledge base"""
        # Check facts
        if query in self.facts:
            return self.facts[query]
        
        # Try to infer using rules
        for rule in self.rules:
            if self._evaluate_condition(rule.condition):
                return self._execute_action(rule.action)
        
        return None
    
    def _evaluate_condition(self, condition: str) -> bool:
        """Evaluate logical condition"""
        # Simple evaluation (can be extended)
        try:
            # Replace fact names with values
            for fact, value in self.facts.items():
                condition = condition.replace(fact, str(value))
            
            # Evaluate
            return eval(condition)
        except:
            return False
    
    def _execute_action(self, action: str) -> Any:
        """Execute action"""
        # Simple action execution
        if action.startswith("set "):
            fact_name = action[4:].split("=")[0].strip()
            fact_value = action.split("=")[1].strip()
            self.add_fact(fact_name, fact_value)
            return fact_value
        return action


class RuleBasedSystem:
    """
    Rule-Based System
    
    Forward and backward chaining inference
    """
    
    def __init__(self):
        self.kb = KnowledgeBase()
        self.working_memory: Dict[str, Any] = {}
    
    def forward_chaining(self, goal: str) -> bool:
        """
        Forward chaining inference
        
        Parameters
        ----------
        goal : str
            Goal to prove
            
        Returns
        -------
        proven : bool
            Whether goal is proven
        """
        agenda = [goal]
        proven = set()
        
        while agenda:
            current = agenda.pop(0)
            
            if current in self.kb.facts:
                proven.add(current)
                continue
            
            # Find rules that can prove current
            for rule in self.kb.rules:
                if rule.action == current:
                    # Check if condition is satisfied
                    if self._check_condition(rule.condition, proven):
                        proven.add(current)
                        # Add condition facts to agenda
                        condition_facts = self._extract_facts(rule.condition)
                        agenda.extend(condition_facts)
                        break
        
        return goal in proven
    
    def backward_chaining(self, goal: str) -> bool:
        """
        Backward chaining inference
        
        Parameters
        ----------
        goal : str
            Goal to prove
            
        Returns
        -------
        proven : bool
            Whether goal is proven
        """
        if goal in self.kb.facts:
            return True
        
        # Find rules that can prove goal
        for rule in self.kb.rules:
            if rule.action == goal:
                # Recursively prove condition
                if self._check_condition(rule.condition, set()):
                    return True
        
        return False
    
    def _check_condition(self, condition: str, proven: set) -> bool:
        """Check if condition is satisfied"""
        # Simple condition checking
        for fact in self.kb.facts:
            if fact in condition and fact in proven:
                condition = condition.replace(fact, "True")
        
        try:
            return eval(condition)
        except:
            return False
    
    def _extract_facts(self, condition: str) -> List[str]:
        """Extract fact names from condition"""
        # Simple extraction
        facts = []
        for fact in self.kb.facts:
            if fact in condition:
                facts.append(fact)
        return facts


class ExpertSystem:
    """
    Expert System
    
    Knowledge-based system for decision making
    """
    
    def __init__(self, domain: str = "general"):
        """
        Initialize expert system
        
        Parameters
        ----------
        domain : str
            Domain of expertise
        """
        self.domain = domain
        self.kb = KnowledgeBase()
        self.rules = RuleBasedSystem()
        self.explanations: List[str] = []
    
    def add_knowledge(self, fact: str, value: Any):
        """Add knowledge to system"""
        self.kb.add_fact(fact, value)
    
    def add_expert_rule(self, condition: str, conclusion: str, confidence: float = 1.0):
        """Add expert rule"""
        self.kb.add_rule(condition, conclusion, confidence)
        self.rules.kb = self.kb
    
    def consult(self, query: str) -> Dict[str, Any]:
        """
        Consult expert system
        
        Parameters
        ----------
        query : str
            Query/question
            
        Returns
        -------
        result : dict
            Consultation result
        """
        # Try forward chaining
        proven = self.rules.forward_chaining(query)
        
        # Try backward chaining
        if not proven:
            proven = self.rules.backward_chaining(query)
        
        # Get explanation
        explanation = self._generate_explanation(query)
        
        return {
            'query': query,
            'answer': proven,
            'confidence': 1.0 if proven else 0.0,
            'explanation': explanation,
            'domain': self.domain
        }
    
    def _generate_explanation(self, query: str) -> str:
        """Generate explanation for result"""
        if query in self.kb.facts:
            return f"Direct fact: {query} = {self.kb.facts[query]}"
        
        # Find applicable rules
        applicable_rules = []
        for rule in self.kb.rules:
            if rule.action == query:
                applicable_rules.append(rule)
        
        if applicable_rules:
            rule = applicable_rules[0]
            return f"Rule: If {rule.condition} then {rule.action}"
        
        return "No explanation available"
