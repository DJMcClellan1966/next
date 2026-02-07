"""
Curriculum: Personalized Learning & Education Platform.
Socratic tutoring, personality-based learning paths, adaptive curriculum,
active learning, knowledge graphs, ethical education.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "socratic", "name": "Socratic Method & Tutoring", "short": "Socratic", "color": "#2563eb"},
    {"id": "adaptive", "name": "Adaptive & Personalized Learning", "short": "Adaptive", "color": "#059669"},
    {"id": "active", "name": "Active Learning & Assessment", "short": "Active", "color": "#7c3aed"},
    {"id": "design", "name": "Learning Design & Analytics", "short": "Design", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    # --- Socratic Method & Tutoring ---
    {"id": "pl_socratic_basics", "book_id": "socratic", "level": "basics",
     "title": "The Socratic Method",
     "learn": "Teaching through questions, not answers. Elicit understanding by guiding learners to discover knowledge themselves. Maieutics: the art of intellectual midwifery.",
     "try_code": "# Socratic loop: ask question → student responds → follow up with deeper question",
     "try_demo": "pl_socratic_dialogue"},
    {"id": "pl_scaffolding", "book_id": "socratic", "level": "intermediate",
     "title": "Scaffolding & Zone of Proximal Development",
     "learn": "Vygotsky's ZPD: the gap between what a learner can do alone vs with guidance. Scaffolding provides temporary support.",
     "try_code": "# Hint levels: none → nudge → partial → worked example",
     "try_demo": "pl_scaffolding"},
    {"id": "pl_feynman", "book_id": "socratic", "level": "advanced",
     "title": "Feynman Technique for Deep Learning",
     "learn": "Explain concept simply → identify gaps → study gaps → simplify again. Teaching to learn. Rubber duck debugging as pedagogy.",
     "try_code": "# Score explanation: simplicity, accuracy, completeness, analogy use",
     "try_demo": "pl_feynman_technique"},
    {"id": "pl_metacognition", "book_id": "socratic", "level": "expert",
     "title": "Metacognitive Strategies",
     "learn": "Thinking about thinking. Self-regulation, planning, monitoring, evaluating. Growth mindset and deliberate practice.",
     "try_code": "# Metacognitive prompt: 'What do I know? What don't I know? How can I find out?'",
     "try_demo": None},

    # --- Adaptive & Personalized Learning ---
    {"id": "pl_learner_model", "book_id": "adaptive", "level": "basics",
     "title": "Learner Modeling",
     "learn": "Track knowledge state: what the learner knows, doesn't know, and partially knows. Bayesian knowledge tracing (BKT).",
     "try_code": "# BKT: P(know|correct) = P(correct|know)*P(know) / P(correct)",
     "try_demo": "pl_knowledge_tracing"},
    {"id": "pl_personality", "book_id": "adaptive", "level": "intermediate",
     "title": "Personality-Based Learning Paths",
     "learn": "MBTI/Big Five → learning style preferences. Visual/auditory/kinesthetic. Adapting content format to learner type.",
     "try_code": "# Match learning style: visual→diagrams, verbal→text, kinesthetic→interactive",
     "try_demo": "pl_personality_match"},
    {"id": "pl_concept_drift", "book_id": "adaptive", "level": "advanced",
     "title": "Concept Drift & Forgetting",
     "learn": "Ebbinghaus forgetting curve: R = e^{-t/S}. Spaced repetition optimizes review timing. Detecting when knowledge fades.",
     "try_code": "import numpy as np\n# Forgetting curve: retention = exp(-time / strength)\ndef retention(t, S): return np.exp(-t/S)",
     "try_demo": "pl_forgetting_curve"},
    {"id": "pl_irt", "book_id": "adaptive", "level": "expert",
     "title": "Item Response Theory (IRT)",
     "learn": "P(correct|θ,a,b,c) = c + (1-c)/(1+exp(-a(θ-b))). θ=ability, a=discrimination, b=difficulty, c=guessing. CAT: computerized adaptive testing.",
     "try_code": "# IRT 3PL: probability of correct response given ability and item params",
     "try_demo": "pl_adaptive_test"},

    # --- Active Learning & Assessment ---
    {"id": "pl_active_basics", "book_id": "active", "level": "basics",
     "title": "Active Learning Strategies",
     "learn": "Think-pair-share, problem-based learning, project-based learning. Engagement > passive consumption. Bloom's taxonomy.",
     "try_code": "# Bloom's levels: remember → understand → apply → analyze → evaluate → create",
     "try_demo": "pl_blooms_classify"},
    {"id": "pl_question_gen", "book_id": "active", "level": "intermediate",
     "title": "Intelligent Question Generation",
     "learn": "Generate questions at appropriate difficulty. Distractor generation for MCQs. Open-ended question templates. Bloom-level targeting.",
     "try_code": "# Generate question at target Bloom level for given concept",
     "try_demo": "pl_question_gen"},
    {"id": "pl_misconception", "book_id": "active", "level": "advanced",
     "title": "Misconception Detection & Repair",
     "learn": "Common misconceptions per topic. Diagnostic questions that reveal specific misunderstandings. Targeted remediation paths.",
     "try_code": "# Detect misconception from wrong answer pattern",
     "try_demo": "pl_misconception_detect"},
    {"id": "pl_portfolio", "book_id": "active", "level": "expert",
     "title": "Portfolio-Based Assessment",
     "learn": "Collect evidence of learning over time. Rubric design. Self-reflection prompts. Competency-based progression.",
     "try_code": "# Portfolio: collect artifacts, rate against rubric, track growth",
     "try_demo": None},

    # --- Learning Design & Analytics ---
    {"id": "pl_curriculum_design", "book_id": "design", "level": "basics",
     "title": "Curriculum Graph Design",
     "learn": "Prerequisite DAG: which concepts must come first? Topological sort for learning order. Shortest path to any goal concept.",
     "try_code": "# DAG: linear_algebra → ML → deep_learning → transformers",
     "try_demo": "pl_prerequisite_graph"},
    {"id": "pl_analytics", "book_id": "design", "level": "intermediate",
     "title": "Learning Analytics",
     "learn": "Track time-on-task, completion rates, error patterns. Predict at-risk learners. A/B testing for pedagogical interventions.",
     "try_code": "# Predict at-risk: low engagement + declining scores → intervention",
     "try_demo": "pl_risk_prediction"},
    {"id": "pl_gamification", "book_id": "design", "level": "advanced",
     "title": "Gamification & Motivation",
     "learn": "Points, badges, streaks, leaderboards. Self-determination theory: autonomy, competence, relatedness. Intrinsic vs extrinsic motivation.",
     "try_code": "# Gamification: award badge when milestone reached",
     "try_demo": "pl_gamification"},
    {"id": "pl_ethical_ed", "book_id": "design", "level": "expert",
     "title": "Ethics in Educational AI",
     "learn": "Student data privacy (FERPA, GDPR). Algorithmic fairness in grading. Transparency in AI tutoring. Avoiding learned helplessness.",
     "try_code": "# Audit: is the AI tutor equally effective across demographics?",
     "try_demo": None},
]


def get_curriculum(): return list(CURRICULUM)
def get_books(): return list(BOOKS)
def get_levels(): return list(LEVELS)
def get_by_book(book_id: str): return [c for c in CURRICULUM if c["book_id"] == book_id]
def get_by_level(level: str): return [c for c in CURRICULUM if c["level"] == level]
def get_item(item_id: str):
    for c in CURRICULUM:
        if c["id"] == item_id: return c
    return None
