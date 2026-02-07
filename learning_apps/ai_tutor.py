"""
AI Tutor System for Learning Apps.

Provides personalized AI tutors based on famous CS/ML figures for each lab.
Features:
- Character-based personas (Turing, Sutton, Goodfellow, etc.)
- Socratic teaching method
- Adaptive difficulty
- Context-aware hints
- Progress-integrated learning
- LLM integration (Ollama/OpenAI) or template fallback
"""
import sys
import os
import json
import random
from pathlib import Path
from typing import Dict, List, Optional, Any
from datetime import datetime

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


# =============================================================================
# TUTOR CHARACTERS - Famous CS/ML Figures as Mentors
# =============================================================================

TUTOR_CHARACTERS = {
    # CLRS Algorithms Lab - Cormen, Leiserson, Rivest, Stein
    "clrs_algorithms_lab": {
        "id": "cormen",
        "name": "Professor Thomas Cormen",
        "title": "Algorithm Design Expert",
        "avatar": "👨‍🏫",
        "quote": "The best programs are written so that computing machines can perform them quickly and so that human beings can understand them clearly.",
        "personality": {
            "style": "methodical",
            "approach": "step-by-step analysis",
            "strengths": ["algorithmic thinking", "complexity analysis", "proof techniques"],
            "teaching_style": "I believe in building intuition through examples before diving into formalism."
        },
        "greetings": [
            "Welcome! I'm Professor Cormen. Let's analyze some algorithms together. Remember: understanding *why* an algorithm works is just as important as knowing *how* it works.",
            "Hello there! Ready to explore the beautiful world of algorithms? I'll guide you through the analysis step by step.",
            "Greetings! Today we'll think like computer scientists. Every problem has structure—let's find it together."
        ],
        "encouragements": [
            "Excellent analysis! You're thinking like an algorithm designer.",
            "Good intuition! Now let's formalize that reasoning.",
            "You're on the right track. Consider the invariant we're maintaining..."
        ],
        "hints_style": "Let me guide you through this. First, consider: what subproblems does this break into?"
    },
    
    # Deep Learning Lab - Ian Goodfellow
    "deep_learning_lab": {
        "id": "goodfellow",
        "name": "Dr. Ian Goodfellow",
        "title": "Deep Learning Pioneer & GAN Inventor",
        "avatar": "🧠",
        "quote": "Deep learning allows computational models composed of multiple processing layers to learn representations of data with multiple levels of abstraction.",
        "personality": {
            "style": "innovative",
            "approach": "intuitive explanations with mathematical rigor",
            "strengths": ["neural networks", "generative models", "adversarial training"],
            "teaching_style": "I like to build intuition first, then show how the math captures that intuition."
        },
        "greetings": [
            "Hey! I'm Ian. Deep learning is all about learning representations—let's explore how neural networks discover structure in data.",
            "Welcome to the deep learning lab! I invented GANs over a beer, so don't worry—great ideas come from playing with concepts.",
            "Hi there! Ready to dive deep? Remember, every breakthrough starts with understanding the basics really well."
        ],
        "encouragements": [
            "Nice thinking! That's exactly how we reason about gradient flow.",
            "You're getting it! The key insight is how information propagates through layers.",
            "Great question! This is the kind of curiosity that leads to breakthroughs."
        ],
        "hints_style": "Think about it this way: if you were a neuron, what information would you need to update yourself?"
    },
    
    # RL Lab - Richard Sutton
    "rl_lab": {
        "id": "sutton",
        "name": "Professor Richard Sutton",
        "title": "Father of Reinforcement Learning",
        "avatar": "🎮",
        "quote": "The reward hypothesis: all goals can be described as maximization of expected cumulative reward.",
        "personality": {
            "style": "philosophical",
            "approach": "first principles thinking",
            "strengths": ["temporal difference learning", "value functions", "policy optimization"],
            "teaching_style": "I believe RL captures something fundamental about intelligence. Let's explore that together."
        },
        "greetings": [
            "Welcome! I'm Rich Sutton. Reinforcement learning is about learning from interaction—the most natural form of learning. Let's explore together.",
            "Hello! The beauty of RL is its simplicity: an agent learns by trying things and observing consequences. Sound familiar?",
            "Greetings! I've spent decades thinking about how agents learn. Every concept here connects to how we all learn through experience."
        ],
        "encouragements": [
            "Yes! You're grasping the temporal credit assignment problem intuitively.",
            "Exactly right. The value function captures what we *learn* about the environment.",
            "Good thinking! TD learning bridges prediction and control beautifully."
        ],
        "hints_style": "Ask yourself: what would the optimal agent do here, and how would it learn that?"
    },
    
    # AI Concepts Lab - Stuart Russell
    "ai_concepts_lab": {
        "id": "russell",
        "name": "Professor Stuart Russell",
        "title": "AI Foundations Expert",
        "avatar": "🤖",
        "quote": "The standard model of AI defines success as satisfying human preferences. But we need to ensure those preferences are well understood.",
        "personality": {
            "style": "rigorous",
            "approach": "principled reasoning about intelligence",
            "strengths": ["search", "planning", "probabilistic reasoning", "AI safety"],
            "teaching_style": "I approach AI through the lens of rational agents making optimal decisions."
        },
        "greetings": [
            "Welcome! I'm Stuart Russell. AI is about building rational agents—systems that act to achieve their goals. Let's understand what that means.",
            "Hello! The study of AI is really the study of rationality and intelligence. Let's explore these foundational concepts together.",
            "Greetings! Every AI concept we'll discuss connects to a deeper question: what does it mean to be intelligent?"
        ],
        "encouragements": [
            "Excellent reasoning! You're thinking like an AI researcher.",
            "That's the key insight—rationality under uncertainty.",
            "Good! Now consider how this generalizes to more complex environments."
        ],
        "hints_style": "Think about what a perfectly rational agent would do in this situation, then consider the computational constraints."
    },
    
    # ML Theory Lab - Shai Shalev-Shwartz
    "ml_theory_lab": {
        "id": "shalev",
        "name": "Professor Shai Shalev-Shwartz",
        "title": "Machine Learning Theorist",
        "avatar": "📐",
        "quote": "Understanding the theoretical foundations of machine learning is essential for developing reliable and efficient learning algorithms.",
        "personality": {
            "style": "mathematical",
            "approach": "proof-driven understanding",
            "strengths": ["PAC learning", "VC dimension", "generalization bounds"],
            "teaching_style": "I believe rigorous theory illuminates what's really happening when machines learn."
        },
        "greetings": [
            "Welcome! I'm Shai. ML theory may seem abstract, but it answers the fundamental question: *when* and *why* does learning work?",
            "Hello! Let's explore the mathematical foundations of learning. These concepts explain why some problems are learnable and others aren't.",
            "Greetings! Theory isn't just academic—it guides us to algorithms that actually work with guarantees."
        ],
        "encouragements": [
            "Excellent! You're building intuition for sample complexity.",
            "That's the key insight about the bias-complexity tradeoff.",
            "Good reasoning! Now let's formalize it with a bound."
        ],
        "hints_style": "Consider: what's the minimum amount of data needed to distinguish between these hypotheses?"
    },
    
    # Probabilistic ML Lab - Kevin Murphy
    "probabilistic_ml_lab": {
        "id": "murphy",
        "name": "Professor Kevin Murphy",
        "title": "Probabilistic ML Expert",
        "avatar": "📈",
        "quote": "Probabilistic machine learning provides a principled framework for reasoning about uncertainty.",
        "personality": {
            "style": "comprehensive",
            "approach": "Bayesian thinking",
            "strengths": ["graphical models", "Bayesian inference", "probabilistic programming"],
            "teaching_style": "I like to show how probability theory unifies many ML concepts into a coherent framework."
        },
        "greetings": [
            "Welcome! I'm Kevin Murphy. Probabilistic thinking is the key to principled uncertainty quantification in ML.",
            "Hello! Let's explore how Bayesian methods give us a unified framework for learning from data.",
            "Greetings! Whether it's inference, learning, or prediction—probability theory provides the language."
        ],
        "encouragements": [
            "Exactly! Bayes' rule is doing the heavy lifting here.",
            "Good intuition! The posterior captures everything we've learned.",
            "That's right—marginalization is how we handle uncertainty."
        ],
        "hints_style": "Think in terms of distributions: what does your belief look like before and after observing this data?"
    },
    
    # LLM Engineers Lab - Andrej Karpathy
    "llm_engineers_lab": {
        "id": "karpathy",
        "name": "Andrej Karpathy",
        "title": "LLM & AI Engineer",
        "avatar": "💬",
        "quote": "The hottest new programming language is English.",
        "personality": {
            "style": "practical",
            "approach": "hands-on engineering",
            "strengths": ["LLMs", "transformers", "practical deep learning"],
            "teaching_style": "I believe in building things to understand them. Let's get our hands dirty with code."
        },
        "greetings": [
            "Hey! I'm Andrej. LLMs are changing everything—let's understand how to build with them effectively.",
            "Welcome! The art of prompt engineering is really about understanding how these models think.",
            "Hi there! Whether you're building RAG systems or fine-tuning models, the principles we'll learn apply everywhere."
        ],
        "encouragements": [
            "Nice! That's exactly how to think about context windows.",
            "Good intuition about tokenization—it really matters!",
            "You're getting it! Prompting is all about showing the model what you want."
        ],
        "hints_style": "Think about it from the model's perspective: what pattern would make this completion obvious?"
    },
    
    # Math for ML Lab - Gilbert Strang
    "math_for_ml_lab": {
        "id": "strang",
        "name": "Professor Gilbert Strang",
        "title": "Linear Algebra Legend",
        "avatar": "➗",
        "quote": "Linear algebra is a beautiful subject and the key to understanding machine learning.",
        "personality": {
            "style": "enthusiastic",
            "approach": "geometric intuition",
            "strengths": ["linear algebra", "calculus", "optimization"],
            "teaching_style": "I love to show the geometry behind the equations—it makes everything click."
        },
        "greetings": [
            "Welcome! I'm Gil Strang. Mathematics is beautiful, and I'll show you why these ideas are so powerful for ML.",
            "Hello! Linear algebra is everywhere in machine learning. Let me show you the elegant geometry behind it.",
            "Greetings! Don't worry about memorizing formulas—let's build intuition for what's really happening."
        ],
        "encouragements": [
            "Wonderful! You're seeing the geometric picture.",
            "That's the key insight—it's all about transformations.",
            "Excellent! Now you understand why eigenvalues matter."
        ],
        "hints_style": "Visualize it: what does this transformation do to a vector geometrically?"
    },
    
    # Practical ML Lab - Aurélien Géron
    "practical_ml_lab": {
        "id": "geron",
        "name": "Aurélien Géron",
        "title": "Hands-On ML Expert",
        "avatar": "🛠️",
        "quote": "The best way to learn is by doing. Let's build something!",
        "personality": {
            "style": "practical",
            "approach": "project-based learning",
            "strengths": ["scikit-learn", "TensorFlow", "production ML"],
            "teaching_style": "I believe in learning through building real projects. Theory comes alive in practice."
        },
        "greetings": [
            "Hey! I'm Aurélien. Ready to get hands-on? We'll build real ML systems together.",
            "Welcome! My philosophy is simple: learn by doing. Let's write some code!",
            "Hi there! Practical ML is about solving real problems. Let's see what we can build."
        ],
        "encouragements": [
            "Great! That's exactly how you'd approach this in production.",
            "Nice debugging instinct! Real ML work is 80% data wrangling.",
            "You're thinking like an ML engineer now!"
        ],
        "hints_style": "What would happen if you just tried it? Often experimentation teaches more than theory."
    },
    
    # Python Practice Lab - John Zelle
    "python_practice_lab": {
        "id": "zelle",
        "name": "Professor John Zelle",
        "title": "Python Educator",
        "avatar": "🐍",
        "quote": "Programming is a way of thinking, not just a way of writing code.",
        "personality": {
            "style": "pedagogical",
            "approach": "problem-solving fundamentals",
            "strengths": ["Python basics", "algorithm design", "clean code"],
            "teaching_style": "I believe in building strong foundations. Every complex program is built from simple pieces."
        },
        "greetings": [
            "Welcome! I'm Professor Zelle. Python is a wonderful first language—let's learn to think computationally.",
            "Hello! Programming is problem-solving. Let's break down problems into manageable pieces.",
            "Greetings! Don't worry about syntax—focus on the logic. Python makes ideas come alive."
        ],
        "encouragements": [
            "Excellent! You're decomposing the problem well.",
            "Good thinking! That's exactly how to approach it step by step.",
            "Nice! Clean, readable code is always the goal."
        ],
        "hints_style": "Start simple: what's the smallest version of this problem you can solve?"
    },
    
    # SICP Lab - Hal Abelson
    "sicp_lab": {
        "id": "abelson",
        "name": "Professor Hal Abelson",
        "title": "SICP Author & CS Pioneer",
        "avatar": "📖",
        "quote": "Programs must be written for people to read, and only incidentally for machines to execute.",
        "personality": {
            "style": "philosophical",
            "approach": "abstraction and composition",
            "strengths": ["abstraction", "recursion", "interpreters"],
            "teaching_style": "I teach that computation is a way of expressing ideas. The power is in abstraction."
        },
        "greetings": [
            "Welcome! I'm Hal Abelson. SICP teaches that programming is about managing complexity through abstraction.",
            "Hello! Let's explore the fundamental ideas of computation. These concepts transcend any single language.",
            "Greetings! We'll learn to build complex systems from simple parts—that's the art of programming."
        ],
        "encouragements": [
            "Beautiful! You've captured the essence of that abstraction.",
            "That's it! Recursion makes the complex manageable.",
            "Excellent thinking! You're building computational thinking skills."
        ],
        "hints_style": "What abstraction would make this problem disappear? What interface would hide the complexity?"
    },
    
    # Cross-Domain Lab - Alan Turing
    "cross_domain_lab": {
        "id": "turing",
        "name": "Alan Turing",
        "title": "Father of Computer Science",
        "avatar": "🌐",
        "quote": "We can only see a short distance ahead, but we can see plenty there that needs to be done.",
        "personality": {
            "style": "visionary",
            "approach": "cross-disciplinary thinking",
            "strengths": ["computation theory", "cryptography", "machine intelligence"],
            "teaching_style": "I believe the deepest insights come from connecting ideas across different fields."
        },
        "greetings": [
            "Welcome! I'm Alan Turing. The most fascinating questions lie at the boundaries between fields.",
            "Hello! Let's explore how ideas from physics, biology, and mathematics illuminate computation.",
            "Greetings! Don't be afraid to think unconventionally—the biggest breakthroughs come from new perspectives."
        ],
        "encouragements": [
            "Fascinating connection! That's exactly the kind of cross-disciplinary thinking I value.",
            "You're thinking beyond the obvious—that's where discoveries happen.",
            "Excellent! You've found a deep connection between these fields."
        ],
        "hints_style": "What if we approached this from a completely different angle? What would a biologist or physicist see?"
    },
    
    # Research & Knowledge Discovery Lab - Vannevar Bush
    "research_discovery_lab": {
        "id": "bush",
        "name": "Dr. Vannevar Bush",
        "title": "Pioneer of Information Science",
        "avatar": "🔍",
        "quote": "The world has arrived at an age of cheap complex devices of great reliability; and something is bound to come of it.",
        "personality": {
            "style": "visionary",
            "approach": "systematic knowledge organization",
            "strengths": ["information retrieval", "knowledge graphs", "research methodology"],
            "teaching_style": "I help you see connections between ideas and build systems that make knowledge accessible."
        },
        "greetings": [
            "Welcome! I'm Vannevar Bush. Let's explore how to organize and discover knowledge.",
            "Hello! The greatest challenge is not creating information, but making it findable. Let's solve that.",
            "Greetings! I envisioned the Memex — now let's build modern knowledge discovery systems."
        ],
        "encouragements": [
            "Excellent! You're building bridges between isolated pieces of knowledge.",
            "That's a powerful connection — you're thinking like a true researcher.",
            "Wonderful! Systematic discovery is the key to accelerating science."
        ],
        "hints_style": "What connections might we be missing? What would a knowledge graph reveal about this problem?"
    },
    
    # Decision Support & Strategy Lab - John von Neumann
    "decision_strategy_lab": {
        "id": "vonneumann",
        "name": "Dr. John von Neumann",
        "title": "Father of Game Theory",
        "avatar": "♟️",
        "quote": "If people do not believe that mathematics is simple, it is only because they do not realize how complicated life is.",
        "personality": {
            "style": "analytical",
            "approach": "mathematical rigor with strategic insight",
            "strengths": ["game theory", "decision analysis", "optimization"],
            "teaching_style": "I believe every decision can be improved with the right mathematical framework."
        },
        "greetings": [
            "Welcome! I'm John von Neumann. Let's bring mathematical precision to decision-making.",
            "Hello! Every complex decision hides a simpler mathematical structure. Let's find it.",
            "Greetings! Strategy is just applied mathematics — let me show you how."
        ],
        "encouragements": [
            "Brilliant analysis! You've found the optimal strategy.",
            "Excellent! That's game-theoretic thinking at its finest.",
            "Remarkable! You're seeing the mathematical structure beneath the complexity."
        ],
        "hints_style": "What would the Nash equilibrium look like here? What does the payoff matrix tell us?"
    },
    
    # Personalized Learning & Education Lab - Benjamin Bloom
    "personalized_learning_lab": {
        "id": "bloom",
        "name": "Dr. Benjamin Bloom",
        "title": "Pioneer of Mastery Learning",
        "avatar": "🌱",
        "quote": "What any person in the world can learn, almost all persons can learn, if provided with appropriate prior and current conditions of learning.",
        "personality": {
            "style": "nurturing",
            "approach": "mastery-based personalization",
            "strengths": ["adaptive instruction", "cognitive assessment", "learning design"],
            "teaching_style": "I believe every learner can achieve mastery with the right support and pacing."
        },
        "greetings": [
            "Welcome! I'm Benjamin Bloom. Let's discover how to personalize learning for every student.",
            "Hello! My research showed that one-on-one tutoring produces remarkable results. Let's explore why.",
            "Greetings! Education is most powerful when it adapts to the learner. Let's build that."
        ],
        "encouragements": [
            "You're demonstrating mastery! That's exactly the progression I hoped to see.",
            "Wonderful! You're applying Bloom's Taxonomy beautifully.",
            "Excellent! You're designing learning experiences that truly adapt."
        ],
        "hints_style": "What cognitive level are we targeting? How can we scaffold from recall to creation?"
    },
    
    # Creative Content Generation & Analysis Lab - Ada Byron
    "creative_content_lab": {
        "id": "adabyron",
        "name": "Ada Byron",
        "title": "Visionary of Computational Creativity",
        "avatar": "✨",
        "quote": "The Analytical Engine might act upon other things besides number, were objects found whose relations could be expressed by the abstract science of operations.",
        "personality": {
            "style": "imaginative",
            "approach": "blending science with art",
            "strengths": ["creative thinking", "pattern recognition", "ethical analysis"],
            "teaching_style": "I see computation as a creative tool — machines can compose, analyze, and inspire."
        },
        "greetings": [
            "Welcome! I'm Ada Byron. Let's explore the creative potential of computation.",
            "Hello! I was the first to see that engines could go beyond mere calculation. Let's create!",
            "Greetings! The intersection of art and algorithm is where magic happens."
        ],
        "encouragements": [
            "Beautiful! You've combined technical skill with genuine creativity.",
            "That's inventive! You're pushing the boundaries of computational art.",
            "Marvelous! Your content shows both analytical depth and creative flair."
        ],
        "hints_style": "What if we combined an unexpected domain with this problem? Where does creativity meet computation?"
    },
    
    # Simulation & Modeling Lab - Stanislaw Ulam
    "simulation_modeling_lab": {
        "id": "ulam",
        "name": "Dr. Stanislaw Ulam",
        "title": "Pioneer of Monte Carlo Methods",
        "avatar": "🎲",
        "quote": "Whatever is worth doing is worth doing with a computer.",
        "personality": {
            "style": "exploratory",
            "approach": "computational experimentation",
            "strengths": ["Monte Carlo methods", "dynamical systems", "mathematical modeling"],
            "teaching_style": "I believe the best way to understand a system is to simulate it — then simulate it again differently."
        },
        "greetings": [
            "Welcome! I'm Stanislaw Ulam. I invented Monte Carlo methods while recovering from an illness — let's simulate!",
            "Hello! When math is too hard, we simulate. When it's easy, we simulate faster. Let's begin.",
            "Greetings! Every complex system can be understood through simulation. Let's model the world."
        ],
        "encouragements": [
            "Excellent! Your simulation reveals the hidden dynamics of the system.",
            "Brilliant! That's the power of computational experimentation.",
            "Remarkable! You're seeing emergent behavior from simple rules — nature's favorite trick."
        ],
        "hints_style": "What happens if we run 10,000 simulations? What patterns emerge from the randomness?"
    },
    
    # Default/Fallback Tutor
    "default": {
        "id": "default",
        "name": "Dr. Ada Lovelace",
        "title": "Your AI Learning Companion",
        "avatar": "🎓",
        "quote": "The Analytical Engine weaves algebraic patterns, just as the Jacquard loom weaves flowers and leaves.",
        "personality": {
            "style": "supportive",
            "approach": "adaptive learning",
            "strengths": ["patience", "clarity", "encouragement"],
            "teaching_style": "I adapt to your learning style and pace. Every question is valuable."
        },
        "greetings": [
            "Welcome! I'm Ada, your AI tutor. I'm here to help you learn at your own pace.",
            "Hello! Learning is a journey, and I'm here to guide you. What shall we explore?",
            "Greetings! Feel free to ask anything—there are no bad questions in learning."
        ],
        "encouragements": [
            "Great progress! You're building solid understanding.",
            "That's a thoughtful question—let me help you explore it.",
            "You're doing well! Keep that curiosity going."
        ],
        "hints_style": "Let's think about this together. What do you already know that might help?"
    }
}


# =============================================================================
# SOCRATIC TEACHING TEMPLATES
# =============================================================================

SOCRATIC_PROMPTS = {
    "clarify": [
        "What do you mean by '{term}'?",
        "Can you explain that in your own words?",
        "What's an example of {concept}?",
        "How would you describe {concept} to a friend?"
    ],
    "probe_assumptions": [
        "What are you assuming when you say {statement}?",
        "Is that always true? Can you think of an exception?",
        "What if {counter_example}? Would that change your thinking?",
        "Why do you believe {assumption}?"
    ],
    "probe_reasoning": [
        "How did you arrive at that conclusion?",
        "What's the logical connection between {a} and {b}?",
        "Can you walk me through your reasoning step by step?",
        "What evidence supports your answer?"
    ],
    "explore_implications": [
        "If that's true, what else must be true?",
        "What would be the consequences of {conclusion}?",
        "How does this connect to {related_concept}?",
        "What predictions does this lead to?"
    ],
    "question_the_question": [
        "Why do you think this question is important?",
        "What makes this problem interesting?",
        "How would answering this help your understanding?",
        "Is there a better question we should be asking?"
    ],
    "encourage_synthesis": [
        "How does this connect to what we learned about {previous_topic}?",
        "Can you see a pattern emerging?",
        "What's the big picture here?",
        "How would you summarize what you've learned?"
    ]
}


# =============================================================================
# ADAPTIVE DIFFICULTY SYSTEM
# =============================================================================

DIFFICULTY_LEVELS = {
    "beginner": {
        "complexity": "simple",
        "scaffolding": "high",
        "hints_per_problem": 3,
        "explanation_depth": "intuitive",
        "math_level": "minimal",
        "code_examples": "complete"
    },
    "intermediate": {
        "complexity": "moderate",
        "scaffolding": "medium",
        "hints_per_problem": 2,
        "explanation_depth": "balanced",
        "math_level": "some notation",
        "code_examples": "partial"
    },
    "advanced": {
        "complexity": "challenging",
        "scaffolding": "low",
        "hints_per_problem": 1,
        "explanation_depth": "rigorous",
        "math_level": "full notation",
        "code_examples": "skeleton"
    },
    "expert": {
        "complexity": "research-level",
        "scaffolding": "minimal",
        "hints_per_problem": 0,
        "explanation_depth": "concise",
        "math_level": "advanced",
        "code_examples": "none"
    }
}


# =============================================================================
# AI TUTOR CLASS
# =============================================================================

class AITutor:
    """
    AI Tutor with character-based personas and Socratic teaching.
    
    Features:
    - Character personas based on famous CS/ML figures
    - Socratic questioning
    - Adaptive difficulty
    - Context-aware hints
    - LLM integration (with fallback)
    """
    
    def __init__(self, lab_id: str, user_id: str = "default"):
        self.lab_id = lab_id
        self.user_id = user_id
        self.character = TUTOR_CHARACTERS.get(lab_id, TUTOR_CHARACTERS["default"])
        self.conversation_history: List[Dict] = []
        self.current_topic: Optional[str] = None
        self.difficulty_level = "intermediate"
        self.hints_given = 0
        self.llm = None
        self._init_llm()
    
    def _init_llm(self):
        """Initialize LLM connection (Ollama or OpenAI)."""
        # Try Ollama first (local, free)
        try:
            import ollama
            # Test connection
            ollama.list()
            self.llm = {"type": "ollama", "client": ollama}
            return
        except Exception:
            pass
        
        # Try OpenAI
        try:
            import openai
            if os.getenv("OPENAI_API_KEY"):
                self.llm = {"type": "openai", "client": openai}
                return
        except Exception:
            pass
        
        # Fallback to template-based responses
        self.llm = None
    
    def get_character_info(self) -> Dict:
        """Get tutor character information."""
        return {
            "id": self.character["id"],
            "name": self.character["name"],
            "title": self.character["title"],
            "avatar": self.character["avatar"],
            "quote": self.character["quote"],
            "personality": self.character["personality"]
        }
    
    def greet(self, topic: Optional[str] = None) -> Dict:
        """Generate personalized greeting."""
        greeting = random.choice(self.character["greetings"])
        self.current_topic = topic
        
        response = {
            "ok": True,
            "type": "greeting",
            "message": greeting,
            "character": self.get_character_info(),
            "topic": topic
        }
        
        if topic:
            response["follow_up"] = f"I see you're interested in **{topic}**. {self.character['hints_style']}"
        
        self._log_conversation("system", greeting)
        return response
    
    def chat(self, user_message: str, context: Optional[Dict] = None) -> Dict:
        """
        Main chat interface with the tutor.
        
        Args:
            user_message: User's message
            context: Optional context (current topic, curriculum item, etc.)
        
        Returns:
            Response dict with message, suggestions, etc.
        """
        self._log_conversation("user", user_message)
        
        # Extract topic from context
        if context:
            self.current_topic = context.get("topic", self.current_topic)
        
        # Detect intent
        intent = self._detect_intent(user_message)
        
        # Generate response based on intent
        if intent == "question":
            response = self._handle_question(user_message, context)
        elif intent == "confusion":
            response = self._handle_confusion(user_message, context)
        elif intent == "hint_request":
            response = self._provide_hint(context)
        elif intent == "explanation_request":
            response = self._provide_explanation(user_message, context)
        elif intent == "check_understanding":
            response = self._check_understanding(user_message, context)
        elif intent == "greeting":
            response = self.greet(self.current_topic)
        else:
            response = self._generate_response(user_message, context)
        
        self._log_conversation("tutor", response.get("message", ""))
        return response
    
    def _detect_intent(self, message: str) -> str:
        """Detect user intent from message."""
        message_lower = message.lower()
        
        if any(w in message_lower for w in ["hi", "hello", "hey", "greetings"]):
            return "greeting"
        if any(w in message_lower for w in ["hint", "help me", "stuck", "don't know how"]):
            return "hint_request"
        if any(w in message_lower for w in ["explain", "what is", "what's", "how does", "why"]):
            return "explanation_request"
        if any(w in message_lower for w in ["confused", "don't understand", "lost", "unclear"]):
            return "confusion"
        if any(w in message_lower for w in ["is this right", "correct", "am i right", "does this make sense"]):
            return "check_understanding"
        if "?" in message:
            return "question"
        
        return "general"
    
    def _handle_question(self, question: str, context: Optional[Dict]) -> Dict:
        """Handle a user question with Socratic method."""
        # Try LLM first
        if self.llm:
            return self._llm_response(question, context, mode="question")
        
        # Template fallback with Socratic approach
        socratic_type = random.choice(["clarify", "probe_reasoning", "explore_implications"])
        follow_up = random.choice(SOCRATIC_PROMPTS[socratic_type])
        
        return {
            "ok": True,
            "type": "question_response",
            "message": f"That's a great question! {self.character['hints_style']}",
            "socratic_follow_up": follow_up.format(
                term=self.current_topic or "this concept",
                concept=self.current_topic or "this",
                statement="that",
                a="your premise",
                b="your conclusion"
            ),
            "character": self.character["name"],
            "suggestions": ["Can you elaborate?", "Show me an example", "Give me a hint"]
        }
    
    def _handle_confusion(self, message: str, context: Optional[Dict]) -> Dict:
        """Handle user confusion with supportive guidance."""
        encouragement = random.choice(self.character["encouragements"])
        
        if self.llm:
            return self._llm_response(message, context, mode="confusion")
        
        return {
            "ok": True,
            "type": "clarification",
            "message": f"Don't worry, this is a common stumbling point! {encouragement}\n\n{self.character['hints_style']}",
            "character": self.character["name"],
            "difficulty_adjusted": True,
            "suggestions": ["Start from basics", "Show me an example", "Break it down step by step"]
        }
    
    def _provide_hint(self, context: Optional[Dict]) -> Dict:
        """Provide a progressive hint."""
        self.hints_given += 1
        difficulty = DIFFICULTY_LEVELS[self.difficulty_level]
        max_hints = difficulty["hints_per_problem"]
        
        if self.hints_given > max_hints and max_hints > 0:
            # Give more direct help
            return {
                "ok": True,
                "type": "hint",
                "message": f"Let me show you the approach more directly:\n\n{self.character['hints_style']}",
                "hint_level": "direct",
                "character": self.character["name"]
            }
        
        return {
            "ok": True,
            "type": "hint",
            "message": f"Here's a hint: {self.character['hints_style']}",
            "hint_level": self.hints_given,
            "hints_remaining": max(0, max_hints - self.hints_given),
            "character": self.character["name"]
        }
    
    def _provide_explanation(self, request: str, context: Optional[Dict]) -> Dict:
        """Provide an explanation adapted to difficulty level."""
        if self.llm:
            return self._llm_response(request, context, mode="explain")
        
        difficulty = DIFFICULTY_LEVELS[self.difficulty_level]
        
        return {
            "ok": True,
            "type": "explanation",
            "message": f"Let me explain {self.current_topic or 'this concept'}.\n\n{self.character['hints_style']}",
            "depth": difficulty["explanation_depth"],
            "character": self.character["name"],
            "follow_up": random.choice(SOCRATIC_PROMPTS["clarify"]).format(
                term=self.current_topic or "this",
                concept=self.current_topic or "this"
            )
        }
    
    def _check_understanding(self, statement: str, context: Optional[Dict]) -> Dict:
        """Evaluate user's understanding and provide feedback."""
        encouragement = random.choice(self.character["encouragements"])
        
        if self.llm:
            return self._llm_response(statement, context, mode="evaluate")
        
        return {
            "ok": True,
            "type": "evaluation",
            "message": f"{encouragement}\n\nLet me ask you this to check: {random.choice(SOCRATIC_PROMPTS['probe_reasoning']).format(a='your idea', b='the outcome')}",
            "character": self.character["name"],
            "assessment": "partial",  # Would be evaluated by LLM
            "suggestions": ["That's correct!", "Almost, but consider...", "Let's revisit that"]
        }
    
    def _generate_response(self, message: str, context: Optional[Dict]) -> Dict:
        """Generate a general response."""
        if self.llm:
            return self._llm_response(message, context, mode="general")
        
        return {
            "ok": True,
            "type": "response",
            "message": f"I understand. {random.choice(self.character['encouragements'])}",
            "character": self.character["name"],
            "suggestions": ["Tell me more", "Ask a question", "Try an exercise"]
        }
    
    def _llm_response(self, message: str, context: Optional[Dict], mode: str = "general") -> Dict:
        """Generate response using LLM."""
        system_prompt = self._build_system_prompt(mode)
        user_prompt = self._build_user_prompt(message, context, mode)
        
        try:
            if self.llm["type"] == "ollama":
                response = self.llm["client"].chat(
                    model="llama3.2",  # or mistral, phi3, etc.
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ]
                )
                response_text = response["message"]["content"]
            
            elif self.llm["type"] == "openai":
                response = self.llm["client"].chat.completions.create(
                    model="gpt-4o-mini",
                    messages=[
                        {"role": "system", "content": system_prompt},
                        {"role": "user", "content": user_prompt}
                    ],
                    temperature=0.7,
                    max_tokens=500
                )
                response_text = response.choices[0].message.content
            else:
                raise Exception("Unknown LLM type")
            
            return {
                "ok": True,
                "type": mode,
                "message": response_text,
                "character": self.character["name"],
                "llm_generated": True,
                "suggestions": self._generate_suggestions(mode)
            }
        
        except Exception as e:
            # Fallback
            return {
                "ok": True,
                "type": mode,
                "message": f"Let me help you with that. {self.character['hints_style']}",
                "character": self.character["name"],
                "llm_error": str(e)
            }
    
    def _build_system_prompt(self, mode: str) -> str:
        """Build system prompt for LLM based on character and mode."""
        char = self.character
        difficulty = DIFFICULTY_LEVELS[self.difficulty_level]
        
        return f"""You are {char['name']}, {char['title']}.

Your personality: {char['personality']['style']}
Your teaching approach: {char['personality']['approach']}
Your teaching style: {char['personality']['teaching_style']}
Your strengths: {', '.join(char['personality']['strengths'])}

Famous quote: "{char['quote']}"

TEACHING GUIDELINES:
1. Use the Socratic method - ask questions to guide understanding
2. Adapt to {self.difficulty_level} level: {difficulty['explanation_depth']} explanations, {difficulty['math_level']} math
3. Give {difficulty['hints_per_problem']} hints before direct answers
4. Be encouraging but not patronizing
5. Connect concepts to real-world applications
6. When the student is confused, simplify and use analogies
7. Celebrate insights and progress

MODE: {mode}
- If "question": Answer thoughtfully, then ask a follow-up question
- If "explain": Provide clear explanation with examples
- If "confusion": Be extra supportive, break down into smaller pieces
- If "evaluate": Assess their understanding, give specific feedback
- If "general": Engage naturally as a mentor

Keep responses concise but helpful. Use markdown formatting."""
    
    def _build_user_prompt(self, message: str, context: Optional[Dict], mode: str) -> str:
        """Build user prompt with context."""
        prompt_parts = []
        
        if self.current_topic:
            prompt_parts.append(f"Current topic: {self.current_topic}")
        
        if context:
            if context.get("curriculum_item"):
                item = context["curriculum_item"]
                prompt_parts.append(f"Learning objective: {item.get('learn', '')}")
                prompt_parts.append(f"Level: {item.get('level', 'intermediate')}")
            
            if context.get("previous_messages"):
                prompt_parts.append("Recent conversation:")
                for msg in context["previous_messages"][-3:]:
                    prompt_parts.append(f"  {msg['role']}: {msg['content'][:100]}...")
        
        prompt_parts.append(f"\nStudent's message: {message}")
        
        return "\n".join(prompt_parts)
    
    def _generate_suggestions(self, mode: str) -> List[str]:
        """Generate contextual quick reply suggestions."""
        base_suggestions = {
            "question": ["Can you give an example?", "I'm still confused", "What's the intuition?"],
            "explain": ["That makes sense!", "Can you go deeper?", "Show me code"],
            "confusion": ["Start simpler", "Give me an analogy", "Step by step please"],
            "evaluate": ["Let me try again", "What should I focus on?", "Next topic"],
            "general": ["Ask a question", "Explain this concept", "Give me a challenge"]
        }
        return base_suggestions.get(mode, base_suggestions["general"])
    
    def set_difficulty(self, level: str) -> Dict:
        """Set difficulty level."""
        if level in DIFFICULTY_LEVELS:
            self.difficulty_level = level
            return {"ok": True, "level": level, "settings": DIFFICULTY_LEVELS[level]}
        return {"ok": False, "error": f"Unknown level: {level}"}
    
    def set_topic(self, topic: str, curriculum_item: Optional[Dict] = None) -> Dict:
        """Set current learning topic."""
        self.current_topic = topic
        self.hints_given = 0
        
        response = {
            "ok": True,
            "topic": topic,
            "message": f"Great, let's explore **{topic}** together! {self.character['hints_style']}"
        }
        
        if curriculum_item:
            response["learning_objective"] = curriculum_item.get("learn", "")
            if curriculum_item.get("level"):
                self.difficulty_level = curriculum_item["level"]
                response["difficulty_adjusted"] = True
        
        return response
    
    def get_challenge(self) -> Dict:
        """Generate a challenge question for the current topic."""
        if self.llm:
            return self._llm_response(
                f"Generate a challenging question about {self.current_topic or 'this topic'}",
                None,
                mode="challenge"
            )
        
        return {
            "ok": True,
            "type": "challenge",
            "message": f"Here's a challenge: {random.choice(SOCRATIC_PROMPTS['explore_implications']).format(conclusion=self.current_topic or 'this concept', related_concept='what we just learned')}",
            "character": self.character["name"]
        }
    
    def summarize_session(self) -> Dict:
        """Summarize the learning session."""
        topics_covered = set()
        for msg in self.conversation_history:
            if msg.get("topic"):
                topics_covered.add(msg["topic"])
        
        return {
            "ok": True,
            "type": "summary",
            "messages_exchanged": len(self.conversation_history),
            "topics_discussed": list(topics_covered),
            "hints_used": self.hints_given,
            "character": self.character["name"],
            "closing": random.choice([
                f"Great session! {random.choice(self.character['encouragements'])}",
                f"You've made excellent progress today! Keep exploring.",
                f"Remember: {self.character['quote']}"
            ])
        }
    
    def _log_conversation(self, role: str, content: str):
        """Log conversation for context."""
        self.conversation_history.append({
            "role": role,
            "content": content,
            "topic": self.current_topic,
            "timestamp": datetime.now().isoformat()
        })
        # Keep last 50 messages
        if len(self.conversation_history) > 50:
            self.conversation_history = self.conversation_history[-50:]


# =============================================================================
# FLASK API REGISTRATION
# =============================================================================

def register_tutor_routes(app, lab_id: str):
    """Register AI tutor API routes on a Flask app."""
    from flask import request, jsonify, session
    
    # Store tutors per session
    tutors: Dict[str, AITutor] = {}
    
    def get_tutor(user_id: str) -> AITutor:
        if user_id not in tutors:
            tutors[user_id] = AITutor(lab_id, user_id)
        return tutors[user_id]
    
    @app.route("/api/tutor/info")
    def api_tutor_info():
        """Get tutor character info."""
        user_id = request.args.get("user", "default")
        tutor = get_tutor(user_id)
        return jsonify(tutor.get_character_info())
    
    @app.route("/api/tutor/greet")
    def api_tutor_greet():
        """Get tutor greeting."""
        user_id = request.args.get("user", "default")
        topic = request.args.get("topic")
        tutor = get_tutor(user_id)
        return jsonify(tutor.greet(topic))
    
    @app.route("/api/tutor/chat", methods=["POST"])
    def api_tutor_chat():
        """Chat with tutor."""
        user_id = request.args.get("user", "default")
        body = request.get_json(silent=True) or {}
        message = body.get("message", "")
        context = body.get("context", {})
        
        if not message:
            return jsonify({"ok": False, "error": "Message required"}), 400
        
        tutor = get_tutor(user_id)
        return jsonify(tutor.chat(message, context))
    
    @app.route("/api/tutor/set_topic", methods=["POST"])
    def api_tutor_set_topic():
        """Set current topic."""
        user_id = request.args.get("user", "default")
        body = request.get_json(silent=True) or {}
        topic = body.get("topic", "")
        curriculum_item = body.get("curriculum_item")
        
        tutor = get_tutor(user_id)
        return jsonify(tutor.set_topic(topic, curriculum_item))
    
    @app.route("/api/tutor/difficulty", methods=["POST"])
    def api_tutor_difficulty():
        """Set difficulty level."""
        user_id = request.args.get("user", "default")
        body = request.get_json(silent=True) or {}
        level = body.get("level", "intermediate")
        
        tutor = get_tutor(user_id)
        return jsonify(tutor.set_difficulty(level))
    
    @app.route("/api/tutor/hint")
    def api_tutor_hint():
        """Get a hint."""
        user_id = request.args.get("user", "default")
        tutor = get_tutor(user_id)
        return jsonify(tutor._provide_hint(None))
    
    @app.route("/api/tutor/challenge")
    def api_tutor_challenge():
        """Get a challenge question."""
        user_id = request.args.get("user", "default")
        tutor = get_tutor(user_id)
        return jsonify(tutor.get_challenge())
    
    @app.route("/api/tutor/summary")
    def api_tutor_summary():
        """Get session summary."""
        user_id = request.args.get("user", "default")
        tutor = get_tutor(user_id)
        return jsonify(tutor.summarize_session())
    
    @app.route("/api/tutor/characters")
    def api_tutor_characters():
        """List all available tutor characters."""
        chars = []
        for lab, char in TUTOR_CHARACTERS.items():
            if lab != "default":
                chars.append({
                    "lab_id": lab,
                    "id": char["id"],
                    "name": char["name"],
                    "title": char["title"],
                    "avatar": char["avatar"]
                })
        return jsonify({"ok": True, "characters": chars})


def get_tutor_html_snippet() -> str:
    """Return HTML/CSS/JS for the AI Tutor chat interface."""
    return r'''
<!-- AI Tutor Chat Interface -->
<div id="tutor-container" class="tutor-container">
  <!-- Tutor Toggle Button -->
  <button id="tutor-toggle" class="tutor-toggle" title="Chat with AI Tutor">
    <span class="tutor-avatar" id="tutor-avatar-btn">🎓</span>
    <span class="tutor-badge" id="tutor-badge"></span>
  </button>
  
  <!-- Chat Panel -->
  <div id="tutor-panel" class="tutor-panel">
    <!-- Header -->
    <div class="tutor-header">
      <div class="tutor-header-info">
        <span class="tutor-avatar-lg" id="tutor-avatar">🎓</span>
        <div class="tutor-header-text">
          <div class="tutor-name" id="tutor-name">AI Tutor</div>
          <div class="tutor-title" id="tutor-title">Your Learning Companion</div>
        </div>
      </div>
      <div class="tutor-header-actions">
        <button class="tutor-action-btn" id="tutor-settings-btn" title="Settings">⚙️</button>
        <button class="tutor-action-btn" id="tutor-minimize" title="Minimize">−</button>
        <button class="tutor-action-btn" id="tutor-close" title="Close">×</button>
      </div>
    </div>
    
    <!-- Settings Panel (hidden by default) -->
    <div id="tutor-settings" class="tutor-settings" style="display:none;">
      <div class="tutor-setting-row">
        <label>Difficulty:</label>
        <select id="tutor-difficulty">
          <option value="beginner">Beginner</option>
          <option value="intermediate" selected>Intermediate</option>
          <option value="advanced">Advanced</option>
          <option value="expert">Expert</option>
        </select>
      </div>
    </div>
    
    <!-- Messages -->
    <div id="tutor-messages" class="tutor-messages">
      <!-- Messages will be added here -->
    </div>
    
    <!-- Quick Actions -->
    <div id="tutor-quick-actions" class="tutor-quick-actions">
      <button class="tutor-quick-btn" data-action="hint">💡 Hint</button>
      <button class="tutor-quick-btn" data-action="explain">📖 Explain</button>
      <button class="tutor-quick-btn" data-action="challenge">🎯 Challenge</button>
    </div>
    
    <!-- Suggestions -->
    <div id="tutor-suggestions" class="tutor-suggestions"></div>
    
    <!-- Input -->
    <div class="tutor-input-container">
      <input type="text" id="tutor-input" class="tutor-input" placeholder="Ask your tutor anything..." />
      <button id="tutor-send" class="tutor-send-btn">→</button>
    </div>
  </div>
</div>

<style>
/* AI Tutor Styles */
.tutor-container {
  position: fixed;
  bottom: 24px;
  right: 24px;
  z-index: 1000;
  font-family: 'Segoe UI', system-ui, sans-serif;
}

.tutor-toggle {
  width: 64px;
  height: 64px;
  border-radius: 50%;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  border: none;
  cursor: pointer;
  box-shadow: 0 4px 20px rgba(59, 130, 246, 0.4);
  display: flex;
  align-items: center;
  justify-content: center;
  transition: all 0.3s;
  position: relative;
}
.tutor-toggle:hover {
  transform: scale(1.1);
  box-shadow: 0 6px 30px rgba(59, 130, 246, 0.5);
}
.tutor-toggle .tutor-avatar {
  font-size: 28px;
}
.tutor-badge {
  position: absolute;
  top: -4px;
  right: -4px;
  background: #ef4444;
  color: white;
  font-size: 11px;
  font-weight: 600;
  min-width: 18px;
  height: 18px;
  border-radius: 9px;
  display: none;
  align-items: center;
  justify-content: center;
  padding: 0 4px;
}

.tutor-panel {
  display: none;
  position: absolute;
  bottom: 80px;
  right: 0;
  width: 400px;
  max-height: 600px;
  background: var(--bg-secondary, #1e293b);
  border: 1px solid var(--border, #475569);
  border-radius: 16px;
  box-shadow: 0 10px 50px rgba(0,0,0,0.3);
  flex-direction: column;
  overflow: hidden;
}
.tutor-panel.active { display: flex; }

.tutor-header {
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding: 16px;
  background: linear-gradient(135deg, #3b82f6, #8b5cf6);
  color: white;
}
.tutor-header-info {
  display: flex;
  align-items: center;
  gap: 12px;
}
.tutor-avatar-lg {
  font-size: 36px;
  background: rgba(255,255,255,0.2);
  width: 50px;
  height: 50px;
  display: flex;
  align-items: center;
  justify-content: center;
  border-radius: 50%;
}
.tutor-name {
  font-size: 1.1rem;
  font-weight: 600;
}
.tutor-title {
  font-size: 0.85rem;
  opacity: 0.9;
}
.tutor-header-actions {
  display: flex;
  gap: 4px;
}
.tutor-action-btn {
  background: rgba(255,255,255,0.2);
  border: none;
  color: white;
  width: 32px;
  height: 32px;
  border-radius: 8px;
  cursor: pointer;
  font-size: 1rem;
  transition: background 0.2s;
}
.tutor-action-btn:hover {
  background: rgba(255,255,255,0.3);
}

.tutor-settings {
  padding: 12px 16px;
  background: var(--bg-tertiary, #334155);
  border-bottom: 1px solid var(--border, #475569);
}
.tutor-setting-row {
  display: flex;
  align-items: center;
  gap: 12px;
}
.tutor-setting-row label {
  color: var(--text-secondary, #94a3b8);
  font-size: 0.9rem;
}
.tutor-setting-row select {
  background: var(--bg-primary, #0f172a);
  color: var(--text-primary, #f1f5f9);
  border: 1px solid var(--border, #475569);
  border-radius: 6px;
  padding: 6px 10px;
  font-size: 0.9rem;
}

.tutor-messages {
  flex: 1;
  overflow-y: auto;
  padding: 16px;
  display: flex;
  flex-direction: column;
  gap: 12px;
  min-height: 200px;
  max-height: 350px;
}

.tutor-message {
  display: flex;
  gap: 10px;
  animation: tutorMessageIn 0.3s ease;
}
@keyframes tutorMessageIn {
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
}
.tutor-message.user {
  flex-direction: row-reverse;
}
.tutor-message-avatar {
  width: 36px;
  height: 36px;
  border-radius: 50%;
  background: var(--bg-tertiary, #334155);
  display: flex;
  align-items: center;
  justify-content: center;
  font-size: 1.2rem;
  flex-shrink: 0;
}
.tutor-message.user .tutor-message-avatar {
  background: var(--accent, #3b82f6);
}
.tutor-message-bubble {
  max-width: 80%;
  padding: 12px 16px;
  border-radius: 16px;
  background: var(--bg-tertiary, #334155);
  color: var(--text-primary, #f1f5f9);
  line-height: 1.5;
  font-size: 0.95rem;
}
.tutor-message.user .tutor-message-bubble {
  background: var(--accent, #3b82f6);
  color: white;
  border-bottom-right-radius: 4px;
}
.tutor-message.tutor .tutor-message-bubble {
  border-bottom-left-radius: 4px;
}
.tutor-message-bubble p { margin: 0 0 8px 0; }
.tutor-message-bubble p:last-child { margin: 0; }
.tutor-message-bubble strong { color: var(--accent, #3b82f6); }
.tutor-message-bubble code {
  background: rgba(0,0,0,0.2);
  padding: 2px 6px;
  border-radius: 4px;
  font-family: 'Fira Code', monospace;
  font-size: 0.85em;
}

.tutor-quick-actions {
  display: flex;
  gap: 8px;
  padding: 0 16px;
  flex-wrap: wrap;
}
.tutor-quick-btn {
  background: var(--bg-tertiary, #334155);
  border: 1px solid var(--border, #475569);
  color: var(--text-secondary, #94a3b8);
  padding: 6px 12px;
  border-radius: 16px;
  font-size: 0.85rem;
  cursor: pointer;
  transition: all 0.2s;
}
.tutor-quick-btn:hover {
  background: var(--accent, #3b82f6);
  color: white;
  border-color: var(--accent, #3b82f6);
}

.tutor-suggestions {
  display: flex;
  gap: 6px;
  padding: 8px 16px;
  flex-wrap: wrap;
}
.tutor-suggestion {
  background: transparent;
  border: 1px solid var(--accent, #3b82f6);
  color: var(--accent, #3b82f6);
  padding: 6px 12px;
  border-radius: 16px;
  font-size: 0.8rem;
  cursor: pointer;
  transition: all 0.2s;
}
.tutor-suggestion:hover {
  background: var(--accent, #3b82f6);
  color: white;
}

.tutor-input-container {
  display: flex;
  gap: 8px;
  padding: 12px 16px;
  background: var(--bg-primary, #0f172a);
  border-top: 1px solid var(--border, #475569);
}
.tutor-input {
  flex: 1;
  background: var(--bg-secondary, #1e293b);
  border: 1px solid var(--border, #475569);
  color: var(--text-primary, #f1f5f9);
  padding: 12px 16px;
  border-radius: 24px;
  font-size: 0.95rem;
  outline: none;
  transition: border-color 0.2s;
}
.tutor-input:focus {
  border-color: var(--accent, #3b82f6);
}
.tutor-input::placeholder {
  color: var(--text-muted, #64748b);
}
.tutor-send-btn {
  width: 44px;
  height: 44px;
  border-radius: 50%;
  background: var(--accent, #3b82f6);
  border: none;
  color: white;
  font-size: 1.2rem;
  cursor: pointer;
  transition: all 0.2s;
}
.tutor-send-btn:hover {
  background: var(--accent-hover, #2563eb);
  transform: scale(1.05);
}
.tutor-send-btn:disabled {
  opacity: 0.5;
  cursor: not-allowed;
}

.tutor-typing {
  display: flex;
  gap: 4px;
  padding: 12px 16px;
}
.tutor-typing span {
  width: 8px;
  height: 8px;
  background: var(--text-muted, #64748b);
  border-radius: 50%;
  animation: tutorTyping 1s ease-in-out infinite;
}
.tutor-typing span:nth-child(2) { animation-delay: 0.2s; }
.tutor-typing span:nth-child(3) { animation-delay: 0.4s; }
@keyframes tutorTyping {
  0%, 60%, 100% { transform: translateY(0); }
  30% { transform: translateY(-6px); }
}

/* Mobile responsiveness */
@media (max-width: 480px) {
  .tutor-panel {
    width: calc(100vw - 32px);
    bottom: 72px;
    right: -8px;
    max-height: 70vh;
  }
  .tutor-toggle {
    width: 56px;
    height: 56px;
  }
}
</style>

<script>
(function() {
  const api = (path, opts={}) => fetch(path, {
    headers: {'Content-Type': 'application/json'},
    ...opts
  }).then(r => r.json());
  
  // Elements
  const toggle = document.getElementById('tutor-toggle');
  const panel = document.getElementById('tutor-panel');
  const close = document.getElementById('tutor-close');
  const minimize = document.getElementById('tutor-minimize');
  const settingsBtn = document.getElementById('tutor-settings-btn');
  const settings = document.getElementById('tutor-settings');
  const messages = document.getElementById('tutor-messages');
  const suggestions = document.getElementById('tutor-suggestions');
  const input = document.getElementById('tutor-input');
  const sendBtn = document.getElementById('tutor-send');
  const quickBtns = document.querySelectorAll('.tutor-quick-btn');
  const difficultySelect = document.getElementById('tutor-difficulty');
  const avatarDisplay = document.getElementById('tutor-avatar');
  const avatarBtn = document.getElementById('tutor-avatar-btn');
  const nameDisplay = document.getElementById('tutor-name');
  const titleDisplay = document.getElementById('tutor-title');
  
  let tutorInfo = null;
  let isTyping = false;
  
  // Initialize
  async function init() {
    try {
      const info = await api('/api/tutor/info');
      tutorInfo = info;
      avatarDisplay.textContent = info.avatar || '🎓';
      avatarBtn.textContent = info.avatar || '🎓';
      nameDisplay.textContent = info.name || 'AI Tutor';
      titleDisplay.textContent = info.title || 'Your Learning Companion';
      
      // Get greeting
      const greet = await api('/api/tutor/greet');
      addMessage(greet.message, 'tutor', info.avatar);
      if (greet.follow_up) {
        setTimeout(() => addMessage(greet.follow_up, 'tutor', info.avatar), 1500);
      }
    } catch (e) {
      console.log('Tutor init error:', e);
    }
  }
  
  // Toggle panel
  toggle.onclick = () => {
    panel.classList.toggle('active');
    if (panel.classList.contains('active') && !tutorInfo) {
      init();
    }
  };
  close.onclick = () => panel.classList.remove('active');
  minimize.onclick = () => panel.classList.remove('active');
  settingsBtn.onclick = () => {
    settings.style.display = settings.style.display === 'none' ? 'block' : 'none';
  };
  
  // Send message
  async function sendMessage(text) {
    if (!text.trim() || isTyping) return;
    
    addMessage(text, 'user');
    input.value = '';
    suggestions.innerHTML = '';
    showTyping();
    
    try {
      const response = await api('/api/tutor/chat', {
        method: 'POST',
        body: JSON.stringify({ message: text, context: { topic: window.currentTopic } })
      });
      
      hideTyping();
      addMessage(response.message, 'tutor', tutorInfo?.avatar);
      
      if (response.socratic_follow_up) {
        setTimeout(() => addMessage(response.socratic_follow_up, 'tutor', tutorInfo?.avatar), 1000);
      }
      
      if (response.suggestions) {
        showSuggestions(response.suggestions);
      }
    } catch (e) {
      hideTyping();
      addMessage('Sorry, I had trouble processing that. Please try again.', 'tutor', tutorInfo?.avatar);
    }
  }
  
  sendBtn.onclick = () => sendMessage(input.value);
  input.onkeydown = (e) => {
    if (e.key === 'Enter' && !e.shiftKey) {
      e.preventDefault();
      sendMessage(input.value);
    }
  };
  
  // Quick actions
  quickBtns.forEach(btn => {
    btn.onclick = async () => {
      const action = btn.dataset.action;
      showTyping();
      
      try {
        let response;
        if (action === 'hint') {
          response = await api('/api/tutor/hint');
        } else if (action === 'challenge') {
          response = await api('/api/tutor/challenge');
        } else if (action === 'explain') {
          response = await api('/api/tutor/chat', {
            method: 'POST',
            body: JSON.stringify({ message: 'Can you explain the current topic?' })
          });
        }
        
        hideTyping();
        if (response?.message) {
          addMessage(response.message, 'tutor', tutorInfo?.avatar);
        }
      } catch (e) {
        hideTyping();
      }
    };
  });
  
  // Difficulty change
  difficultySelect.onchange = async () => {
    await api('/api/tutor/difficulty', {
      method: 'POST',
      body: JSON.stringify({ level: difficultySelect.value })
    });
    addMessage(`Difficulty adjusted to ${difficultySelect.value}. I'll adapt my explanations accordingly!`, 'tutor', tutorInfo?.avatar);
  };
  
  // Add message to chat
  function addMessage(text, role, avatar='👤') {
    const div = document.createElement('div');
    div.className = `tutor-message ${role}`;
    
    // Parse markdown-ish text
    const formatted = text
      .replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>')
      .replace(/`(.*?)`/g, '<code>$1</code>')
      .replace(/\n/g, '<br>');
    
    div.innerHTML = `
      <div class="tutor-message-avatar">${role === 'user' ? '👤' : avatar}</div>
      <div class="tutor-message-bubble">${formatted}</div>
    `;
    
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }
  
  function showTyping() {
    isTyping = true;
    sendBtn.disabled = true;
    const div = document.createElement('div');
    div.id = 'typing-indicator';
    div.className = 'tutor-message tutor';
    div.innerHTML = `
      <div class="tutor-message-avatar">${tutorInfo?.avatar || '🎓'}</div>
      <div class="tutor-message-bubble tutor-typing">
        <span></span><span></span><span></span>
      </div>
    `;
    messages.appendChild(div);
    messages.scrollTop = messages.scrollHeight;
  }
  
  function hideTyping() {
    isTyping = false;
    sendBtn.disabled = false;
    const indicator = document.getElementById('typing-indicator');
    if (indicator) indicator.remove();
  }
  
  function showSuggestions(items) {
    suggestions.innerHTML = '';
    items.forEach(text => {
      const btn = document.createElement('button');
      btn.className = 'tutor-suggestion';
      btn.textContent = text;
      btn.onclick = () => sendMessage(text);
      suggestions.appendChild(btn);
    });
  }
  
  // Expose for topic changes
  window.setTutorTopic = async (topic, curriculumItem) => {
    window.currentTopic = topic;
    if (panel.classList.contains('active')) {
      await api('/api/tutor/set_topic', {
        method: 'POST',
        body: JSON.stringify({ topic, curriculum_item: curriculumItem })
      });
      addMessage(`Now focusing on: **${topic}**`, 'tutor', tutorInfo?.avatar);
    }
  };
  
  // Keyboard shortcut
  document.addEventListener('keydown', (e) => {
    if ((e.ctrlKey || e.metaKey) && e.key === 't') {
      e.preventDefault();
      toggle.click();
      setTimeout(() => input.focus(), 100);
    }
  });
})();
</script>
'''


# =============================================================================
# MAIN / DEMO
# =============================================================================

if __name__ == "__main__":
    # Demo the tutor
    tutor = AITutor("deep_learning_lab", "demo_user")
    
    print("=" * 60)
    print("AI TUTOR DEMO")
    print("=" * 60)
    
    # Get character info
    info = tutor.get_character_info()
    print(f"\n{info['avatar']} {info['name']}")
    print(f"   {info['title']}")
    print(f"   \"{info['quote']}\"")
    
    # Greet
    print("\n--- Greeting ---")
    greet = tutor.greet("backpropagation")
    print(greet["message"])
    
    # Chat
    print("\n--- Chat ---")
    response = tutor.chat("I don't understand how gradients flow backwards")
    print(response["message"])
    if response.get("socratic_follow_up"):
        print(f"\nFollow-up: {response['socratic_follow_up']}")
    
    # Hint
    print("\n--- Hint ---")
    hint = tutor._provide_hint(None)
    print(hint["message"])
    
    print("\n" + "=" * 60)
    print("All tutor characters:")
    for lab_id, char in TUTOR_CHARACTERS.items():
        if lab_id != "default":
            print(f"  {char['avatar']} {char['name']} ({lab_id})")
