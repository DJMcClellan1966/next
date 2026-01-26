"""
Socratic Method - Inspired by Socrates

Implements:
- Question-Based Learning
- Interactive Debugging
- Explanation Generation
- Active Learning
- Dialectical Reasoning
"""
import numpy as np
from typing import List, Dict, Any, Optional, Callable, Tuple
import logging
from collections import deque

logger = logging.getLogger(__name__)


class SocraticQuestioner:
    """
    Socratic Questioner - Generates questions to expose contradictions and refine knowledge
    """
    
    def __init__(
        self,
        knowledge_base: Optional[Dict[str, Any]] = None,
        question_types: Optional[List[str]] = None
    ):
        """
        Initialize Socratic Questioner
        
        Args:
            knowledge_base: Base knowledge to question
            question_types: Types of questions to generate
        """
        self.knowledge_base = knowledge_base or {}
        self.question_types = question_types or [
            'clarification', 'assumption', 'evidence', 'implication', 'alternative'
        ]
        self.question_history = []
        self.contradictions_found = []
    
    def generate_question(
        self,
        statement: str,
        context: Optional[Dict[str, Any]] = None,
        question_type: Optional[str] = None
    ) -> str:
        """
        Generate a Socratic question
        
        Args:
            statement: Statement to question
            context: Additional context
            question_type: Specific type of question
        
        Returns:
            Generated question
        """
        if question_type is None:
            question_type = np.random.choice(self.question_types)
        
        questions = {
            'clarification': [
                f"What do you mean by '{statement}'?",
                f"Can you clarify what '{statement}' means?",
                f"How would you define '{statement}'?"
            ],
            'assumption': [
                f"What assumptions are you making about '{statement}'?",
                f"What must be true for '{statement}' to hold?",
                f"What are you assuming when you say '{statement}'?"
            ],
            'evidence': [
                f"What evidence supports '{statement}'?",
                f"How do you know that '{statement}' is true?",
                f"What proof do you have for '{statement}'?"
            ],
            'implication': [
                f"What are the implications of '{statement}'?",
                f"If '{statement}' is true, what else must be true?",
                f"What follows from '{statement}'?"
            ],
            'alternative': [
                f"Are there alternatives to '{statement}'?",
                f"Could '{statement}' be false?",
                f"What if '{statement}' were different?"
            ]
        }
        
        question = np.random.choice(questions.get(question_type, questions['clarification']))
        self.question_history.append({
            'statement': statement,
            'question': question,
            'type': question_type,
            'context': context
        })
        
        return question
    
    def elenchus(self, claim: str, premises: List[str]) -> Dict[str, Any]:
        """
        Elenchus: Refutation through questioning
        
        Exposes contradictions in reasoning
        
        Args:
            claim: Claim to examine
            premises: Premises supporting the claim
        
        Returns:
            Analysis with contradictions found
        """
        contradictions = []
        
        # Check for logical contradictions
        for i, premise1 in enumerate(premises):
            for premise2 in premises[i+1:]:
                if self._contradicts(premise1, premise2):
                    contradictions.append({
                        'premise1': premise1,
                        'premise2': premise2,
                        'type': 'logical_contradiction'
                    })
        
        # Check claim against premises
        for premise in premises:
            if self._contradicts(claim, premise):
                contradictions.append({
                    'claim': claim,
                    'premise': premise,
                    'type': 'claim_premise_contradiction'
                })
        
        self.contradictions_found.extend(contradictions)
        
        return {
            'claim': claim,
            'premises': premises,
            'contradictions': contradictions,
            'valid': len(contradictions) == 0
        }
    
    def _contradicts(self, statement1: str, statement2: str) -> bool:
        """Check if two statements contradict each other (simplified)"""
        # Simple keyword-based contradiction detection
        negations = {
            'is': 'is not',
            'are': 'are not',
            'has': 'has not',
            'can': 'cannot',
            'will': 'will not',
            'true': 'false',
            'yes': 'no'
        }
        
        statement1_lower = statement1.lower()
        statement2_lower = statement2.lower()
        
        for word, negation in negations.items():
            if word in statement1_lower and negation in statement2_lower:
                return True
            if negation in statement1_lower and word in statement2_lower:
                return True
        
        return False
    
    def maieutics(
        self,
        topic: str,
        initial_knowledge: Optional[Dict[str, Any]] = None,
        max_questions: int = 10
    ) -> Dict[str, Any]:
        """
        Maieutics: Drawing out knowledge through questions
        
        Args:
            topic: Topic to explore
            initial_knowledge: Starting knowledge
            max_questions: Maximum questions to ask
        
        Returns:
            Refined knowledge through questioning
        """
        knowledge = initial_knowledge or {}
        questions_asked = []
        
        for i in range(max_questions):
            # Generate question based on current knowledge
            if not knowledge:
                question = f"What do you know about '{topic}'?"
            else:
                # Question an assumption or seek clarification
                key = np.random.choice(list(knowledge.keys()))
                question = self.generate_question(
                    f"{key}: {knowledge[key]}",
                    question_type='assumption'
                )
            
            questions_asked.append(question)
            
            # Simulate answer (in real system, would get from user/model)
            # For now, mark as explored
            knowledge[f'explored_{i}'] = f"Question: {question}"
        
        return {
            'topic': topic,
            'refined_knowledge': knowledge,
            'questions_asked': questions_asked
        }


class SocraticDebugger:
    """
    Socratic Debugger - Question-based error diagnosis
    """
    
    def __init__(self, model: Any):
        """
        Initialize Socratic Debugger
        
        Args:
            model: Model to debug
        """
        self.model = model
        self.questioner = SocraticQuestioner()
        self.debug_history = []
    
    def diagnose_error(
        self,
        error_message: str,
        context: Optional[Dict[str, Any]] = None
    ) -> Dict[str, Any]:
        """
        Diagnose error through Socratic questioning
        
        Args:
            error_message: Error message
            context: Additional context
        
        Returns:
            Diagnosis with questions and findings
        """
        diagnosis = {
            'error': error_message,
            'questions': [],
            'findings': [],
            'hypotheses': []
        }
        
        # Question 1: What is the error?
        q1 = self.questioner.generate_question(
            error_message,
            question_type='clarification'
        )
        diagnosis['questions'].append(q1)
        diagnosis['findings'].append("Error identified: " + error_message)
        
        # Question 2: What assumptions led to this?
        q2 = self.questioner.generate_question(
            f"The model failed with: {error_message}",
            question_type='assumption'
        )
        diagnosis['questions'].append(q2)
        
        # Generate hypotheses
        hypotheses = [
            "Input data format incorrect",
            "Model not properly trained",
            "Missing dependencies",
            "Resource constraints exceeded",
            "Configuration error"
        ]
        diagnosis['hypotheses'] = hypotheses
        
        # Question 3: What evidence supports each hypothesis?
        for hypothesis in hypotheses[:3]:  # Top 3
            q3 = self.questioner.generate_question(
                hypothesis,
                question_type='evidence'
            )
            diagnosis['questions'].append(q3)
        
        self.debug_history.append(diagnosis)
        
        return diagnosis
    
    def interactive_debug(self, error_message: str) -> List[str]:
        """
        Interactive debugging session
        
        Args:
            error_message: Error to debug
        
        Returns:
            List of questions to ask
        """
        questions = []
        
        # Initial diagnosis
        diagnosis = self.diagnose_error(error_message)
        questions.extend(diagnosis['questions'])
        
        # Follow-up questions based on hypotheses
        for hypothesis in diagnosis['hypotheses']:
            follow_up = self.questioner.generate_question(
                f"If {hypothesis} is true",
                question_type='implication'
            )
            questions.append(follow_up)
        
        return questions


class SocraticExplainer:
    """
    Socratic Explainer - Generate explanations through dialogue
    """
    
    def __init__(self, model: Any):
        """
        Initialize Socratic Explainer
        
        Args:
            model: Model to explain
        """
        self.model = model
        self.questioner = SocraticQuestioner()
        self.explanation_dialogue = []
    
    def explain_prediction(
        self,
        prediction: Any,
        input_data: Any,
        max_questions: int = 5
    ) -> Dict[str, Any]:
        """
        Explain prediction through Socratic dialogue
        
        Args:
            prediction: Model prediction
            input_data: Input that led to prediction
            max_questions: Maximum questions in dialogue
        
        Returns:
            Explanation dialogue
        """
        dialogue = []
        
        # Initial statement
        initial = f"The model predicted: {prediction}"
        dialogue.append({'speaker': 'system', 'text': initial})
        
        # Generate questions to explore the prediction
        for i in range(max_questions):
            if i == 0:
                question = self.questioner.generate_question(
                    initial,
                    question_type='clarification'
                )
            elif i == 1:
                question = self.questioner.generate_question(
                    f"Why did the model predict {prediction}?",
                    question_type='evidence'
                )
            else:
                question = self.questioner.generate_question(
                    f"Prediction: {prediction}",
                    question_type=np.random.choice(['assumption', 'implication', 'alternative'])
                )
            
            dialogue.append({'speaker': 'questioner', 'text': question})
            
            # Simulate answer (in real system, would generate from model)
            answer = f"Based on the input features, the model determined that {prediction} is the most likely outcome."
            dialogue.append({'speaker': 'system', 'text': answer})
        
        self.explanation_dialogue = dialogue
        
        return {
            'prediction': prediction,
            'input': str(input_data)[:100],  # Truncate
            'dialogue': dialogue,
            'explanation': self._synthesize_explanation(dialogue)
        }
    
    def _synthesize_explanation(self, dialogue: List[Dict[str, str]]) -> str:
        """Synthesize final explanation from dialogue"""
        system_responses = [d['text'] for d in dialogue if d['speaker'] == 'system']
        return " ".join(system_responses)


class SocraticActiveLearner:
    """
    Socratic Active Learner - Select most informative questions/samples
    """
    
    def __init__(
        self,
        uncertainty_estimator: Optional[Callable] = None
    ):
        """
        Initialize Socratic Active Learner
        
        Args:
            uncertainty_estimator: Function to estimate uncertainty
        """
        self.uncertainty_estimator = uncertainty_estimator
        self.selected_samples = []
        self.questions_asked = []
    
    def select_questions(
        self,
        unlabeled_data: np.ndarray,
        model: Any,
        n_questions: int = 5,
        strategy: str = 'uncertainty'
    ) -> List[int]:
        """
        Select most informative questions (samples to label)
        
        Args:
            unlabeled_data: Unlabeled data
            model: Current model
            n_questions: Number of questions to select
            strategy: Selection strategy
        
        Returns:
            Indices of selected samples
        """
        if strategy == 'uncertainty':
            # Select samples with highest uncertainty
            if hasattr(model, 'predict_proba'):
                probabilities = model.predict_proba(unlabeled_data)
                uncertainties = 1 - np.max(probabilities, axis=1)
            else:
                # Fallback: random
                uncertainties = np.random.random(len(unlabeled_data))
            
            selected_indices = np.argsort(uncertainties)[-n_questions:][::-1]
        
        elif strategy == 'diversity':
            # Select diverse samples
            from sklearn.metrics.pairwise import cosine_similarity
            similarities = cosine_similarity(unlabeled_data)
            # Select samples that are least similar to already selected
            selected_indices = []
            for _ in range(n_questions):
                if len(selected_indices) == 0:
                    selected_indices.append(np.random.randint(len(unlabeled_data)))
                else:
                    # Find sample least similar to selected
                    similarities_to_selected = similarities[:, selected_indices].min(axis=1)
                    selected_indices.append(np.argmax(similarities_to_selected))
        
        else:  # random
            selected_indices = np.random.choice(
                len(unlabeled_data),
                size=min(n_questions, len(unlabeled_data)),
                replace=False
            ).tolist()
        
        self.selected_samples.extend(selected_indices)
        return selected_indices
    
    def generate_question_for_sample(
        self,
        sample: np.ndarray,
        model: Any
    ) -> str:
        """
        Generate a question about a specific sample
        
        Args:
            sample: Data sample
            model: Current model
        
        Returns:
            Question about the sample
        """
        if hasattr(model, 'predict'):
            prediction = model.predict(sample.reshape(1, -1))[0]
            question = f"What is the label for this sample? (Model predicts: {prediction})"
        else:
            question = "What is the label for this sample?"
        
        self.questions_asked.append(question)
        return question
