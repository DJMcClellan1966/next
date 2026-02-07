"""Demos for Personalized Learning & Education Lab."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np


def pl_socratic_dialogue():
    """Demonstrate Socratic method in tutoring."""
    try:
        dialogue = [
            ("Tutor", "What happens when you multiply a matrix by a vector?"),
            ("Student", "You get another vector?"),
            ("Tutor", "Yes! Can you say what that vector represents geometrically?"),
            ("Student", "Hmm... the original vector but transformed?"),
            ("Tutor", "Exactly! Now, what kind of transformations can matrices represent?"),
            ("Student", "Rotation? Scaling?"),
            ("Tutor", "Good! And what about a matrix that only scales a vector without changing its direction?"),
            ("Student", "Oh — that's an eigenvector! And the scale factor is the eigenvalue!"),
            ("Tutor", "Brilliant! You discovered eigenvalues through reasoning, not memorization. 🎉"),
        ]

        out = "Socratic Dialogue: Discovering Eigenvalues\n" + "=" * 50 + "\n\n"
        for speaker, text in dialogue:
            prefix = "  🎓" if speaker == "Tutor" else "  👤"
            out += f"{prefix} [{speaker}]: {text}\n"
        out += "\nKey: The tutor never directly stated what eigenvalues are."
        out += "\nThe student discovered the concept through guided questions."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_scaffolding():
    """Demonstrate scaffolded hint levels."""
    try:
        problem = "Implement binary search on a sorted array"
        scaffolds = [
            ("Level 0 — No hint", "Try it yourself!"),
            ("Level 1 — Nudge", "Binary search eliminates half the array each step. What do you compare?"),
            ("Level 2 — Structure", "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n        mid = ???\n        # Compare arr[mid] with target..."),
            ("Level 3 — Worked Example",
             "def binary_search(arr, target):\n    lo, hi = 0, len(arr)-1\n    while lo <= hi:\n"
             "        mid = (lo + hi) // 2\n        if arr[mid] == target: return mid\n"
             "        elif arr[mid] < target: lo = mid + 1\n        else: hi = mid - 1\n    return -1"),
        ]

        out = f"Scaffolded Hints: {problem}\n" + "=" * 50 + "\n\n"
        for level, hint in scaffolds:
            out += f"  [{level}]\n    {hint}\n\n"
        out += "The tutor starts at Level 0 and only progresses if the student is stuck."
        out += "\nThis matches Vygotsky's Zone of Proximal Development."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_feynman_technique():
    """Demonstrate the Feynman technique for learning."""
    try:
        concept = "Backpropagation"
        stages = [
            ("Step 1: Explain Simply",
             "Backprop calculates how much each weight contributed to the error,\n"
             "    then adjusts weights proportionally. It works backwards from the output."),
            ("Step 2: Identify Gaps",
             "Gap detected: Student couldn't explain WHY we go backwards.\n"
             "    Gap: chain rule application unclear."),
            ("Step 3: Study & Fill Gaps",
             "Chain rule: df/dx = df/dy × dy/dx. Going backwards = applying chain rule\n"
             "    layer by layer: ∂L/∂w = ∂L/∂a × ∂a/∂z × ∂z/∂w"),
            ("Step 4: Simplify Again",
             "Backprop is just the chain rule from calculus applied layer by layer.\n"
             "    Each layer passes blame backwards: 'you caused this much error.'"),
        ]

        out = f"Feynman Technique: {concept}\n" + "=" * 50 + "\n\n"
        for step, text in stages:
            out += f"  [{step}]\n    {text}\n\n"

        # Score the explanation
        scores = {"Simplicity": 8, "Accuracy": 7, "Completeness": 6, "Analogy": 9}
        out += "Explanation Score:\n"
        for metric, score in scores.items():
            bar = "█" * score + "░" * (10 - score)
            out += f"  {metric:>14s}: {bar} {score}/10\n"
        out += "\nOverall: 7.5/10 — Good! Simplify the chain rule explanation further."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_knowledge_tracing():
    """Bayesian Knowledge Tracing simulation."""
    try:
        # BKT parameters
        p_init = 0.1   # Initial probability of knowing
        p_learn = 0.2  # Probability of learning per opportunity
        p_slip = 0.1   # P(wrong | know)
        p_guess = 0.25 # P(correct | don't know)

        responses = [0, 0, 1, 1, 0, 1, 1, 1, 1, 1]  # 0=wrong, 1=correct
        p_know = p_init

        out = "Bayesian Knowledge Tracing\n" + "=" * 50 + "\n"
        out += f"P(init)={p_init}, P(learn)={p_learn}, P(slip)={p_slip}, P(guess)={p_guess}\n\n"
        out += f"{'Step':>4s} {'Response':>8s} {'P(know)':>10s} {'Status':>12s}\n"
        out += f"{'-'*4} {'-'*8} {'-'*10} {'-'*12}\n"

        for i, resp in enumerate(responses):
            # Update
            if resp == 1:
                p_correct_know = (1 - p_slip) * p_know
                p_correct_not = p_guess * (1 - p_know)
                p_know_given_correct = p_correct_know / (p_correct_know + p_correct_not)
                p_know = p_know_given_correct + (1 - p_know_given_correct) * p_learn
            else:
                p_wrong_know = p_slip * p_know
                p_wrong_not = (1 - p_guess) * (1 - p_know)
                p_know_given_wrong = p_wrong_know / (p_wrong_know + p_wrong_not)
                p_know = p_know_given_wrong + (1 - p_know_given_wrong) * p_learn

            status = "Mastered" if p_know > 0.95 else ("Learning" if p_know > 0.5 else "Struggling")
            symbol = "✓" if resp else "✗"
            out += f"  {i+1:>2d}   {symbol:>8s} {p_know:>10.3f} {status:>12s}\n"

        out += f"\nFinal P(know) = {p_know:.3f}"
        out += "\nModel tracks knowledge state from response patterns."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_personality_match():
    """Match learning style to content format."""
    try:
        learner_profiles = [
            {"name": "Alex", "type": "Visual-Intuitive", "preferences": ["diagrams", "concept maps", "videos"],
             "recommendation": "Use flowcharts for algorithms, 3D plots for math, video walkthroughs for code"},
            {"name": "Sam", "type": "Verbal-Sequential", "preferences": ["text", "step-by-step", "pseudocode"],
             "recommendation": "Textbook-style explanations, numbered steps, written proofs"},
            {"name": "Jordan", "type": "Kinesthetic-Global", "preferences": ["interactive", "experiments", "projects"],
             "recommendation": "Hands-on coding exercises, sandbox experiments, build projects"},
        ]

        out = "Personality-Based Learning Paths\n" + "=" * 50 + "\n\n"
        topic = "Understanding Neural Networks"
        out += f"Topic: {topic}\n\n"
        for p in learner_profiles:
            out += f"  👤 {p['name']} ({p['type']})\n"
            out += f"     Preferences: {', '.join(p['preferences'])}\n"
            out += f"     → {p['recommendation']}\n\n"
        out += "Same content, different delivery — personalization improves retention by ~25%."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_forgetting_curve():
    """Ebbinghaus forgetting curve and spaced repetition."""
    try:
        # R(t) = e^(-t/S) where S = stability (higher = slower forgetting)
        reviews = [0, 1, 3, 7, 14, 30]  # Review at these days
        S = 2.0  # Initial stability

        out = "Forgetting Curve & Spaced Repetition\n" + "=" * 50 + "\n"
        out += "R(t) = e^(-t/S), S increases with each review\n\n"

        for review_num, day in enumerate(reviews):
            S_curr = S * (1.5 ** review_num)  # Stability grows with reviews
            out += f"  Review #{review_num} (day {day:>2d}, S={S_curr:.1f}):\n"
            for d in range(min(day + 7, 35)):
                if d < day:
                    continue
                t = d - day
                R = np.exp(-t / S_curr)
                bar = "█" * int(R * 30)
                marker = " ← review" if d in reviews and d > day else ""
                out += f"    Day {d:>2d}: {R:.0%} {bar}{marker}\n"
                if R < 0.5 and d not in reviews:
                    break
            out += "\n"

        out += "Key: Each review strengthens memory (increases S)."
        out += "\nOptimal spacing prevents forgetting while minimizing reviews."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_adaptive_test():
    """Computerized Adaptive Testing with IRT."""
    try:
        np.random.seed(42)
        # Item bank: (difficulty, discrimination)
        items = [
            {"id": "Q1", "b": -2.0, "a": 1.0, "text": "What is 2+2?"},
            {"id": "Q2", "b": -1.0, "a": 1.2, "text": "What is a variable?"},
            {"id": "Q3", "b": 0.0, "a": 1.5, "text": "Explain recursion"},
            {"id": "Q4", "b": 1.0, "a": 1.3, "text": "Implement quicksort"},
            {"id": "Q5", "b": 2.0, "a": 1.1, "text": "Prove NP-completeness"},
            {"id": "Q6", "b": 3.0, "a": 0.9, "text": "Design a distributed system"},
        ]

        def irt_prob(theta, a, b):
            return 1.0 / (1.0 + np.exp(-a * (theta - b)))

        # Simulate adaptive test
        true_theta = 0.5  # Student's true ability
        theta_est = 0.0   # Initial estimate
        used = set()

        out = "Computerized Adaptive Test (CAT)\n" + "=" * 50 + "\n"
        out += f"Student true ability: θ={true_theta}\n\n"

        for step in range(4):
            # Select most informative item (closest difficulty to current estimate)
            best_item = min(
                [it for it in items if it["id"] not in used],
                key=lambda it: abs(it["b"] - theta_est)
            )
            used.add(best_item["id"])

            # Simulate response
            p = irt_prob(true_theta, best_item["a"], best_item["b"])
            correct = np.random.random() < p

            # Update ability estimate (simplified)
            if correct:
                theta_est += 0.5 / (step + 1)
            else:
                theta_est -= 0.5 / (step + 1)

            symbol = "✓" if correct else "✗"
            out += f"  Step {step+1}: {best_item['text']:>30s} (b={best_item['b']:+.1f}) → {symbol}  θ̂={theta_est:+.2f}\n"

        out += f"\nFinal estimate: θ̂={theta_est:+.2f} (true: {true_theta:+.2f})"
        out += f"\nUsed {len(used)}/6 items — CAT adapts difficulty to the student."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_blooms_classify():
    """Classify questions by Bloom's taxonomy level."""
    try:
        questions = [
            ("What is a linked list?", "Remember", "Recall definition"),
            ("Explain how a hash table handles collisions", "Understand", "Explain mechanism"),
            ("Use binary search to find element in sorted array", "Apply", "Implement algorithm"),
            ("Compare quicksort vs mergesort trade-offs", "Analyze", "Compare/contrast"),
            ("Which sorting algorithm is best for this dataset?", "Evaluate", "Judge/select"),
            ("Design a new data structure for this problem", "Create", "Design original solution"),
        ]

        out = "Bloom's Taxonomy Classification\n" + "=" * 50 + "\n\n"
        levels = ["Remember", "Understand", "Apply", "Analyze", "Evaluate", "Create"]
        for i, level in enumerate(levels):
            out += f"  {'▓' * (i+1)}{'░' * (6-i-1)} Level {i+1}: {level}\n"
        out += "\n"
        for question, level, reasoning in questions:
            idx = levels.index(level)
            out += f"  [{level:>10s}] {question}\n"
            out += f"             → {reasoning}\n"
        out += "\nHigher-order questions (Analyze+) promote deeper learning."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_question_gen():
    """Generate questions at specified Bloom's level."""
    try:
        concept = "Binary Search Tree"
        generated = {
            "Remember": "What is the property that defines a BST?",
            "Understand": "Why does BST search run in O(log n) on balanced trees?",
            "Apply": "Insert the values [5, 3, 7, 1, 4] into an empty BST.",
            "Analyze": "What causes a BST to degrade to O(n) search time?",
            "Evaluate": "Should you use a BST or a hash table for this use case? Justify.",
            "Create": "Design a self-balancing BST variant for your application.",
        }

        out = f"Question Generation: {concept}\n" + "=" * 50 + "\n\n"
        for level, question in generated.items():
            out += f"  [{level:>10s}] {question}\n"
        out += "\nEach question targets a specific cognitive level."
        out += "\nThe tutor selects based on learner's current Bloom level."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_misconception_detect():
    """Detect misconceptions from answer patterns."""
    try:
        misconceptions = [
            {
                "question": "What is the time complexity of binary search?",
                "wrong_answer": "O(n)",
                "misconception": "Confusing binary search with linear search",
                "remediation": "Review: binary search eliminates HALF the array each step → log₂(n) steps",
            },
            {
                "question": "In gradient descent, what happens with a very large learning rate?",
                "wrong_answer": "Faster convergence",
                "misconception": "Assuming bigger steps always = faster learning",
                "remediation": "Demo: show overshooting — large η causes oscillation/divergence",
            },
            {
                "question": "What does P(A|B) mean?",
                "wrong_answer": "P(A) × P(B)",
                "misconception": "Confusing conditional probability with independence",
                "remediation": "Review Bayes' theorem: P(A|B) = P(B|A)P(A)/P(B) ≠ P(A)P(B) in general",
            },
        ]

        out = "Misconception Detection & Repair\n" + "=" * 50 + "\n\n"
        for m in misconceptions:
            out += f"  Q: {m['question']}\n"
            out += f"  Wrong: {m['wrong_answer']}\n"
            out += f"  🔍 Misconception: {m['misconception']}\n"
            out += f"  💊 Remediation: {m['remediation']}\n\n"
        out += "Pattern: specific wrong answers reveal specific misunderstandings."
        out += "\nTargeted remediation is more effective than generic review."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_prerequisite_graph():
    """Build and analyze a prerequisite graph."""
    try:
        prereqs = {
            "Transformers": ["Attention", "Neural Networks"],
            "Attention": ["Linear Algebra", "Neural Networks"],
            "Neural Networks": ["Linear Algebra", "Calculus", "Python"],
            "GANs": ["Neural Networks", "Probability"],
            "RL": ["Probability", "Calculus", "Python"],
            "Linear Algebra": ["Math Basics"],
            "Calculus": ["Math Basics"],
            "Probability": ["Math Basics"],
            "Python": [],
            "Math Basics": [],
        }

        # Topological sort
        visited = set()
        order = []
        def topo(node):
            if node in visited:
                return
            visited.add(node)
            for dep in prereqs.get(node, []):
                topo(dep)
            order.append(node)

        for node in prereqs:
            topo(node)

        # Path to Transformers
        def path_to(goal, current_known=set()):
            path = []
            def collect(node):
                if node in current_known or node in path:
                    return
                for dep in prereqs.get(node, []):
                    collect(dep)
                path.append(node)
            collect(goal)
            return path

        learning_path = path_to("Transformers")

        out = "Prerequisite Graph & Learning Path\n" + "=" * 50 + "\n\n"
        out += "Dependency graph:\n"
        for node, deps in prereqs.items():
            if deps:
                out += f"  {node} ← {', '.join(deps)}\n"
            else:
                out += f"  {node} (no prerequisites)\n"
        out += f"\nOptimal path to 'Transformers':\n  "
        out += " → ".join(learning_path)
        out += f"\n\n{len(learning_path)} concepts in dependency order."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_risk_prediction():
    """Predict at-risk learners from engagement data."""
    try:
        np.random.seed(42)
        learners = [
            {"name": "Alice", "login_freq": 0.9, "completion": 0.85, "quiz_avg": 0.78, "trend": "stable"},
            {"name": "Bob", "login_freq": 0.3, "completion": 0.40, "quiz_avg": 0.45, "trend": "declining"},
            {"name": "Carol", "login_freq": 0.7, "completion": 0.65, "quiz_avg": 0.72, "trend": "improving"},
            {"name": "Dave", "login_freq": 0.1, "completion": 0.15, "quiz_avg": 0.30, "trend": "declining"},
            {"name": "Eve", "login_freq": 0.8, "completion": 0.90, "quiz_avg": 0.92, "trend": "stable"},
        ]

        out = "At-Risk Learner Prediction\n" + "=" * 50 + "\n\n"
        out += f"{'Name':>8s} {'Login':>6s} {'Done':>6s} {'Quiz':>6s} {'Trend':>10s} {'Risk':>8s}\n"
        out += f"{'-'*8} {'-'*6} {'-'*6} {'-'*6} {'-'*10} {'-'*8}\n"

        for l in learners:
            risk_score = (1 - l["login_freq"]) * 0.3 + (1 - l["completion"]) * 0.3 + (1 - l["quiz_avg"]) * 0.2
            if l["trend"] == "declining":
                risk_score += 0.2
            risk = "🔴 HIGH" if risk_score > 0.5 else ("🟡 MED" if risk_score > 0.3 else "🟢 LOW")
            out += f"  {l['name']:>6s} {l['login_freq']:>6.0%} {l['completion']:>6.0%} {l['quiz_avg']:>6.0%} {l['trend']:>10s} {risk:>8s}\n"

        out += "\n🔴 HIGH risk: Dave, Bob → Trigger intervention (email, tutor check-in)"
        out += "\n🟢 LOW risk: Alice, Eve → On track"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def pl_gamification():
    """Gamification system demonstration."""
    try:
        badges = [
            {"name": "First Steps", "icon": "🌱", "condition": "Complete first lesson", "earned": True},
            {"name": "Streak Master", "icon": "🔥", "condition": "7-day streak", "earned": True},
            {"name": "Quiz Ace", "icon": "⭐", "condition": "Score 100% on any quiz", "earned": True},
            {"name": "Deep Diver", "icon": "🤿", "condition": "Complete advanced topic", "earned": False},
            {"name": "Mentor", "icon": "🎓", "condition": "Help another learner", "earned": False},
            {"name": "Completionist", "icon": "🏆", "condition": "Complete all topics in a lab", "earned": False},
        ]

        out = "Gamification System\n" + "=" * 50 + "\n\n"
        out += "Badges:\n"
        for b in badges:
            status = "✅ Earned" if b["earned"] else "🔒 Locked"
            out += f"  {b['icon']} {b['name']:>16s} — {b['condition']:>35s} [{status}]\n"

        out += "\nProgress: 3/6 badges earned (50%)\n"
        out += "XP: 1,250 | Level: 5 | Streak: 12 days 🔥\n"
        out += "\nNext milestone: 'Deep Diver' — complete an advanced topic to unlock!"
        out += "\n\nDesign principle: badges for mastery, not just participation."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


# --- Demo dispatcher ---
DEMO_HANDLERS = {
    "pl_socratic_dialogue": pl_socratic_dialogue,
    "pl_scaffolding": pl_scaffolding,
    "pl_feynman_technique": pl_feynman_technique,
    "pl_knowledge_tracing": pl_knowledge_tracing,
    "pl_personality_match": pl_personality_match,
    "pl_forgetting_curve": pl_forgetting_curve,
    "pl_adaptive_test": pl_adaptive_test,
    "pl_blooms_classify": pl_blooms_classify,
    "pl_question_gen": pl_question_gen,
    "pl_misconception_detect": pl_misconception_detect,
    "pl_prerequisite_graph": pl_prerequisite_graph,
    "pl_risk_prediction": pl_risk_prediction,
    "pl_gamification": pl_gamification,
}

def run_demo(demo_id: str) -> dict:
    handler = DEMO_HANDLERS.get(demo_id)
    if handler:
        return handler()
    return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
