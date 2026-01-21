"""
Recommendation Engine Using Adaptive Neuron
Real-world example: Product/content recommendations
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

from quantum_kernel import get_kernel, KernelConfig
from ai.adaptive_neuron import AdaptiveNeuron


class RecommendationEngine:
    """Smart recommendation system using adaptive neuron"""
    
    def __init__(self):
        self.kernel = get_kernel(KernelConfig(use_sentence_transformers=True))
        self.neuron = AdaptiveNeuron(self.kernel, name="Recommender")
        self.items = {}  # item_id -> description
    
    def add_item(self, item_id: str, description: str):
        """Add item to catalog"""
        self.items[item_id] = description
        print(f"  Added item: {item_id} - {description[:50]}...")
    
    def record_user_action(self, user_action: str, item_id: str, liked: bool = True):
        """Record user interaction"""
        if item_id not in self.items:
            return
        
        item_description = self.items[item_id]
        reward = 1.0 if liked else -0.5
        
        # Learn: user action -> item preference
        self.neuron.learn(user_action, item_description, reward)
        print(f"  Learned: '{user_action}' -> {'likes' if liked else 'dislikes'} '{item_description[:40]}...'")
    
    def recommend(self, user_action: str, top_k: int = 5) -> list:
        """Get recommendations based on user action"""
        # Activate neuron to find related items
        result = self.neuron.activate(user_action)
        
        # Find items similar to learned preferences
        recommendations = []
        for item_id, description in self.items.items():
            # Check if item matches learned preferences
            similarity = self.kernel.similarity(description, user_action)
            
            # Boost if neuron learned this association
            if description in result.get('related_concepts', {}):
                similarity += 0.2
            
            recommendations.append((item_id, description, similarity))
        
        # Sort and return top K
        recommendations.sort(key=lambda x: x[2], reverse=True)
        return recommendations[:top_k]


def demo_recommendation_engine():
    """Demonstrate recommendation engine"""
    print("="*70)
    print("RECOMMENDATION ENGINE DEMONSTRATION")
    print("="*70)
    
    # Create engine
    engine = RecommendationEngine()
    
    # Add items (products, content, etc.)
    print("\n[Adding Items to Catalog]")
    engine.add_item("item1", "Python programming course")
    engine.add_item("item2", "Machine learning tutorial")
    engine.add_item("item3", "Data science book")
    engine.add_item("item4", "Web development guide")
    engine.add_item("item5", "JavaScript tutorial")
    engine.add_item("item6", "Deep learning course")
    
    # Learn from user behavior
    print("\n[Learning from User Behavior]")
    engine.record_user_action("user watched Python video", "item1", liked=True)
    engine.record_user_action("user bought ML book", "item2", liked=True)
    engine.record_user_action("user skipped web dev", "item4", liked=False)
    engine.record_user_action("user completed data course", "item3", liked=True)
    
    # Get recommendations
    print("\n[Getting Recommendations]")
    user_action = "user interested in programming"
    recommendations = engine.recommend(user_action, top_k=3)
    
    print(f"\n  User action: '{user_action}'")
    print(f"  Top Recommendations:")
    for i, (item_id, description, score) in enumerate(recommendations, 1):
        print(f"    {i}. [{score:.3f}] {description}")
    
    # More learning
    print("\n[More Learning]")
    engine.record_user_action("user searched deep learning", "item6", liked=True)
    
    # Updated recommendations
    print("\n[Updated Recommendations]")
    recommendations = engine.recommend("user wants to learn AI", top_k=3)
    print(f"  Top Recommendations:")
    for i, (item_id, description, score) in enumerate(recommendations, 1):
        print(f"    {i}. [{score:.3f}] {description}")
    
    # Statistics
    print("\n[Engine Statistics]")
    stats = engine.neuron.get_stats()
    print(f"  Learned associations: {stats['learned_concepts']}")
    print(f"  Total weights: {stats['total_weights']}")
    print(f"  Success rate: {stats['success_rate']:.2%}")
    
    print("\n" + "="*70)
    print("DEMONSTRATION COMPLETE")
    print("="*70 + "\n")


if __name__ == "__main__":
    try:
        demo_recommendation_engine()
    except Exception as e:
        print(f"\nError: {e}")
        import traceback
        traceback.print_exc()
