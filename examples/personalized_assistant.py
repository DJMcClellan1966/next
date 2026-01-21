"""
Personalized AI Assistant Using Adaptive Neuron
Real-world example: Assistant that learns your preferences
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_kernel import get_kernel, KernelConfig
from ai.adaptive_neuron import AdaptiveNeuron


class PersonalizedAssistant:
    """AI Assistant that learns your preferences"""
    
    def __init__(self, user_name: str = "User"):
        self.user_name = user_name
        self.kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
        self.neuron = AdaptiveNeuron(self.kernel, name=f"{user_name}Assistant")
        
        print(f"[+] Personalized Assistant for {user_name} initialized")
    
    def learn_preference(self, preference: str, behavior: str):
        """Learn a user preference"""
        self.neuron.learn(preference, behavior, reward=1.0)
        print(f"  Learned: {preference} -> {behavior}")
    
    def learn_dislike(self, dislike: str, avoid_behavior: str):
        """Learn what user dislikes"""
        self.neuron.learn(dislike, avoid_behavior, reward=-0.5)
        print(f"  Learned to avoid: {dislike} -> {avoid_behavior}")
    
    def ask(self, question: str) -> dict:
        """Ask assistant a question"""
        result = self.neuron.activate(question)
        
        print(f"\n  Q: {question}")
        print(f"  A: {result['response']}")
        print(f"  Confidence: {result['confidence']:.2f}")
        
        return result
    
    def give_feedback(self, question: str, was_helpful: bool):
        """Give feedback to improve assistant"""
        self.neuron.reinforce(question, was_correct=was_helpful)
        print(f"  Feedback recorded: {'Helpful' if was_helpful else 'Not helpful'}")


def demo_personalized_assistant():
    """Demonstrate personalized assistant"""
    print("="*70)
    print("PERSONALIZED AI ASSISTANT DEMONSTRATION")
    print("="*70)
    
    # Create assistant
    assistant = PersonalizedAssistant("John")
    
    # Learn preferences
    print("\n[Learning User Preferences]")
    assistant.learn_preference("I prefer Python", "suggest Python solutions")
    assistant.learn_preference("I like code examples", "always include code")
    assistant.learn_preference("I want short answers", "keep responses concise")
    assistant.learn_dislike("I hate long explanations", "avoid verbose responses")
    assistant.learn_preference("I work with data science", "focus on data analysis")
    
    # Use assistant
    print("\n[Using Assistant]")
    assistant.ask("How do I process data?")
    assistant.ask("What's the best way to learn programming?")
    assistant.ask("Explain machine learning")
    
    # Give feedback
    print("\n[Giving Feedback]")
    assistant.give_feedback("How do I process data?", was_helpful=True)
    assistant.give_feedback("What's the best way to learn programming?", was_helpful=False)
    
    # Assistant improves
    print("\n[Assistant After Feedback]")
    assistant.ask("How do I process data?")  # Should be better now
    
    # Show stats
    print("\n[Assistant Statistics]")
    stats = assistant.neuron.get_stats()
    print(f"  Total interactions: {stats['total_activations']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    print(f"  Learned concepts: {stats['learned_concepts']}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        demo_personalized_assistant()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
