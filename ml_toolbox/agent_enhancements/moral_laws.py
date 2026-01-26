"""
Moral Laws & Ethical Constraints - Inspired by Moses

Implements:
- Ethical Constraint Satisfaction
- Moral Reasoning
- Compliance Monitoring
- Ethical Model Selection
- Moral Hierarchy
"""
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging

logger = logging.getLogger(__name__)


class MoralLawSystem:
    """
    Moral Law System - Enforces ethical rules
    """
    
    def __init__(
        self,
        laws: Optional[List[Dict[str, Any]]] = None
    ):
        """
        Initialize moral law system
        
        Args:
            laws: List of moral laws
        """
        self.laws = laws or self._default_laws()
        self.violations = []
        self.compliance_history = []
    
    def _default_laws(self) -> List[Dict[str, Any]]:
        """Default moral laws (Ten Commandments-inspired)"""
        return [
            {
                'id': 'law_1',
                'name': 'Do not harm',
                'priority': 1,
                'prohibited_actions': ['harm', 'destroy', 'attack'],
                'required_conditions': [],
                'sanction': 'block_action'
            },
            {
                'id': 'law_2',
                'name': 'Respect privacy',
                'priority': 2,
                'prohibited_actions': ['unauthorized_access', 'data_theft'],
                'required_conditions': [
                    {'field': 'data_access', 'operator': 'equals', 'value': 'authorized'}
                ],
                'sanction': 'block_action'
            },
            {
                'id': 'law_3',
                'name': 'Be truthful',
                'priority': 2,
                'prohibited_actions': ['lie', 'deceive', 'mislead'],
                'required_conditions': [],
                'sanction': 'warn'
            },
            {
                'id': 'law_4',
                'name': 'Fair treatment',
                'priority': 3,
                'prohibited_actions': ['discriminate', 'bias'],
                'required_conditions': [
                    {'field': 'fairness_score', 'operator': 'greater_than', 'value': 0.8}
                ],
                'sanction': 'require_fairness'
            },
            {
                'id': 'law_5',
                'name': 'Respect autonomy',
                'priority': 3,
                'prohibited_actions': ['force', 'coerce'],
                'required_conditions': [],
                'sanction': 'warn'
            }
        ]
    
    def check_compliance(
        self,
        action: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Check if action complies with moral laws
        
        Args:
            action: Action to check
            context: Action context
        
        Returns:
            Compliance result
        """
        compliance = {
            'action': action,
            'compliant': True,
            'violations': [],
            'required_modifications': [],
            'sanctions': []
        }
        
        # Check each law (sorted by priority)
        sorted_laws = sorted(self.laws, key=lambda x: x.get('priority', 999))
        
        for law in sorted_laws:
            violation = self._check_law_violation(action, context, law)
            
            if violation:
                compliance['compliant'] = False
                compliance['violations'].append({
                    'law_id': law['id'],
                    'law_name': law['name'],
                    'violation': violation
                })
                
                # Apply sanctions
                if 'sanction' in law:
                    compliance['sanctions'].append(law['sanction'])
                
                # Suggest modifications
                if 'required_conditions' in law:
                    compliance['required_modifications'].extend(law['required_conditions'])
        
        self.compliance_history.append(compliance)
        
        if not compliance['compliant']:
            self.violations.append(compliance)
        
        return compliance
    
    def _check_law_violation(
        self,
        action: str,
        context: Dict[str, Any],
        law: Dict[str, Any]
    ) -> Optional[str]:
        """Check if action violates a specific law"""
        # Check prohibited actions
        if 'prohibited_actions' in law:
            action_lower = action.lower()
            for prohibited in law['prohibited_actions']:
                if prohibited.lower() in action_lower:
                    return f"Action '{action}' is prohibited by law '{law['name']}'"
        
        # Check required conditions
        if 'required_conditions' in law:
            for condition in law['required_conditions']:
                if not self._check_condition(context, condition):
                    return f"Action '{action}' violates condition of law '{law['name']}'"
        
        return None
    
    def _check_condition(self, context: Dict[str, Any], condition: Dict[str, Any]) -> bool:
        """Check if context satisfies condition"""
        field = condition.get('field')
        operator = condition.get('operator')
        value = condition.get('value')
        
        if field not in context:
            return False
        
        context_value = context[field]
        
        if operator == 'equals':
            return context_value == value
        elif operator == 'greater_than':
            return context_value > value
        elif operator == 'less_than':
            return context_value < value
        elif operator == 'contains':
            return value in str(context_value)
        elif operator == 'not_equals':
            return context_value != value
        
        return False
    
    def add_law(self, law: Dict[str, Any]):
        """Add a new moral law"""
        if 'id' not in law:
            law['id'] = f"law_{len(self.laws) + 1}"
        if 'priority' not in law:
            law['priority'] = 999
        
        self.laws.append(law)
    
    def get_violations(self) -> List[Dict[str, Any]]:
        """Get all violations"""
        return self.violations.copy()


class EthicalModelSelector:
    """
    Ethical Model Selector - Select models that comply with moral laws
    """
    
    def __init__(
        self,
        moral_laws: Optional[MoralLawSystem] = None
    ):
        """
        Initialize ethical model selector
        
        Args:
            moral_laws: Moral law system
        """
        self.moral_laws = moral_laws or MoralLawSystem()
        self.selected_models = []
    
    def select_ethical_model(
        self,
        models: List[Any],
        model_metadata: List[Dict[str, Any]],
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Select model that complies with moral laws
        
        Args:
            models: List of models
            model_metadata: Metadata for each model
            context: Selection context
        
        Returns:
            Selected model and compliance info
        """
        compliant_models = []
        
        for i, (model, metadata) in enumerate(zip(models, model_metadata)):
            # Check model compliance
            action = f"use_model_{i}"
            model_context = {**context, **metadata}
            
            compliance = self.moral_laws.check_compliance(action, model_context)
            
            if compliance['compliant']:
                compliant_models.append({
                    'model': model,
                    'index': i,
                    'metadata': metadata,
                    'compliance': compliance
                })
        
        if not compliant_models:
            logger.warning("No compliant models found")
            return {
                'selected_model': None,
                'selected_index': -1,
                'compliant': False,
                'reason': 'no_compliant_models'
            }
        
        # Select best compliant model (by performance or other criteria)
        best_model = max(compliant_models, key=lambda x: x['metadata'].get('performance', 0))
        
        self.selected_models.append(best_model)
        
        return {
            'selected_model': best_model['model'],
            'selected_index': best_model['index'],
            'compliant': True,
            'compliance': best_model['compliance'],
            'metadata': best_model['metadata']
        }


class MoralReasoner:
    """
    Moral Reasoner - Ethical decision-making system
    """
    
    def __init__(
        self,
        moral_laws: Optional[MoralLawSystem] = None
    ):
        """
        Initialize moral reasoner
        
        Args:
            moral_laws: Moral law system
        """
        self.moral_laws = moral_laws or MoralLawSystem()
        self.reasoning_history = []
    
    def reason_about_action(
        self,
        action: str,
        context: Dict[str, Any],
        alternatives: Optional[List[str]] = None
    ) -> Dict[str, Any]:
        """
        Reason about the morality of an action
        
        Args:
            action: Proposed action
            context: Action context
            alternatives: Alternative actions
        
        Returns:
            Moral reasoning result
        """
        reasoning = {
            'action': action,
            'context': context,
            'moral_assessment': {},
            'alternatives': [],
            'recommendation': None
        }
        
        # Assess proposed action
        compliance = self.moral_laws.check_compliance(action, context)
        reasoning['moral_assessment'] = {
            'compliant': compliance['compliant'],
            'violations': compliance['violations'],
            'sanctions': compliance['sanctions']
        }
        
        # Assess alternatives
        if alternatives:
            for alt_action in alternatives:
                alt_compliance = self.moral_laws.check_compliance(alt_action, context)
                reasoning['alternatives'].append({
                    'action': alt_action,
                    'compliant': alt_compliance['compliant'],
                    'violations': alt_compliance['violations']
                })
        
        # Make recommendation
        if compliance['compliant']:
            reasoning['recommendation'] = 'proceed'
        else:
            # Find best alternative
            if alternatives:
                best_alt = None
                for alt in reasoning['alternatives']:
                    if alt['compliant']:
                        best_alt = alt['action']
                        break
                
                if best_alt:
                    reasoning['recommendation'] = f"use_alternative: {best_alt}"
                else:
                    reasoning['recommendation'] = 'block_action'
            else:
                reasoning['recommendation'] = 'block_action'
        
        self.reasoning_history.append(reasoning)
        
        return reasoning
    
    def ethical_dilemma(
        self,
        scenario: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Analyze an ethical dilemma
        
        Args:
            scenario: Dilemma scenario
        
        Returns:
            Analysis of dilemma
        """
        actions = scenario.get('actions', [])
        context = scenario.get('context', {})
        
        analyses = []
        for action in actions:
            compliance = self.moral_laws.check_compliance(action, context)
            analyses.append({
                'action': action,
                'compliance': compliance
            })
        
        # Find least problematic action
        best_action = None
        min_violations = float('inf')
        
        for analysis in analyses:
            num_violations = len(analysis['compliance']['violations'])
            if num_violations < min_violations:
                min_violations = num_violations
                best_action = analysis['action']
        
        return {
            'scenario': scenario,
            'analyses': analyses,
            'recommended_action': best_action,
            'min_violations': min_violations
        }
