"""
Curriculum: Creative Content Generation & Analysis Platform.
LLM-based generation, quality analysis, personality matching,
Socratic ideation, creative exploration, ethical review.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "generation", "name": "Content Generation", "short": "Generate", "color": "#2563eb"},
    {"id": "analysis", "name": "Quality & Style Analysis", "short": "Analyze", "color": "#059669"},
    {"id": "creativity", "name": "Creative AI Techniques", "short": "Creative", "color": "#7c3aed"},
    {"id": "ethics", "name": "Ethics & Responsible AI Content", "short": "Ethics", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    # --- Content Generation ---
    {"id": "cc_prompting", "book_id": "generation", "level": "basics",
     "title": "Prompt Engineering Fundamentals",
     "learn": "Zero-shot, few-shot, chain-of-thought prompting. System/user/assistant roles. Temperature and top-p sampling.",
     "try_code": "# Prompt template: system + context + instruction + output format",
     "try_demo": "cc_prompt_patterns"},
    {"id": "cc_structured", "book_id": "generation", "level": "intermediate",
     "title": "Structured Content Generation",
     "learn": "Generate articles, summaries, reports with structure. Outline-first approach. Section-by-section generation with consistency.",
     "try_code": "# Pipeline: topic → outline → expand sections → review → finalize",
     "try_demo": "cc_article_gen"},
    {"id": "cc_storytelling", "book_id": "generation", "level": "advanced",
     "title": "AI-Assisted Storytelling",
     "learn": "Character arcs, plot structures (3-act, hero's journey). AI as co-author: suggest, don't dictate. Maintaining narrative consistency.",
     "try_code": "# Story structure: setup → rising action → climax → resolution",
     "try_demo": "cc_story_gen"},
    {"id": "cc_multimodal", "book_id": "generation", "level": "expert",
     "title": "Multimodal Content Creation",
     "learn": "Text-to-image prompting (DALL-E, Midjourney), text-to-audio, text-to-video. Cross-modal consistency and style transfer.",
     "try_code": "# Multimodal pipeline: text → image prompt → audio script → video storyboard",
     "try_demo": None},

    # --- Quality & Style Analysis ---
    {"id": "cc_readability", "book_id": "analysis", "level": "basics",
     "title": "Readability & Clarity Analysis",
     "learn": "Flesch-Kincaid, Gunning Fog, SMOG indices. Sentence length, syllable count, passive voice detection. Plain language principles.",
     "try_code": "# Flesch-Kincaid: 206.835 - 1.015*(words/sents) - 84.6*(syllables/words)",
     "try_demo": "cc_readability"},
    {"id": "cc_sentiment", "book_id": "analysis", "level": "intermediate",
     "title": "Sentiment & Tone Analysis",
     "learn": "Positive/negative/neutral classification. Emotion detection (joy, anger, surprise). Tone matching for brand voice.",
     "try_code": "# Simple sentiment: count positive vs negative words",
     "try_demo": "cc_sentiment"},
    {"id": "cc_style_transfer", "book_id": "analysis", "level": "advanced",
     "title": "Style Analysis & Transfer",
     "learn": "Author fingerprinting: vocabulary richness, sentence patterns, punctuation habits. Style transfer: rewrite in different voice.",
     "try_code": "# Style metrics: avg sentence length, vocabulary diversity, formality score",
     "try_demo": "cc_style_analysis"},
    {"id": "cc_factcheck", "book_id": "analysis", "level": "expert",
     "title": "Fact-Checking & Hallucination Detection",
     "learn": "Cross-reference claims with knowledge base. Detect unsupported assertions. Confidence scoring for generated facts.",
     "try_code": "# Verify: extract claims → search knowledge base → score support",
     "try_demo": None},

    # --- Creative AI Techniques ---
    {"id": "cc_brainstorm", "book_id": "creativity", "level": "basics",
     "title": "AI-Powered Brainstorming",
     "learn": "Divergent thinking: generate many ideas. Convergent thinking: filter and combine. SCAMPER method with AI assistance.",
     "try_code": "# SCAMPER: Substitute, Combine, Adapt, Modify, Put to other use, Eliminate, Reverse",
     "try_demo": "cc_brainstorm"},
    {"id": "cc_analogy", "book_id": "creativity", "level": "intermediate",
     "title": "Analogy & Metaphor Generation",
     "learn": "Structure mapping theory. Cross-domain analogies for explanation. Metaphor as a tool for understanding abstract concepts.",
     "try_code": "# Analogy: 'A neural network is like a brain because both have connected nodes'",
     "try_demo": "cc_analogy_gen"},
    {"id": "cc_remix", "book_id": "creativity", "level": "advanced",
     "title": "Creative Remixing & Mashups",
     "learn": "Combine concepts from different domains. Constraint-based creativity. Random stimulus technique. Oblique strategies.",
     "try_code": "# Random mashup: concept_A + concept_B → novel idea",
     "try_demo": "cc_remix"},
    {"id": "cc_evolution", "book_id": "creativity", "level": "expert",
     "title": "Evolutionary Creative Systems",
     "learn": "Genetic algorithms for creative content. Fitness = novelty + quality. Interactive evolutionary art. Surprise as a metric.",
     "try_code": "# Evolve: population → fitness → select → crossover → mutate → repeat",
     "try_demo": None},

    # --- Ethics & Responsible AI Content ---
    {"id": "cc_copyright", "book_id": "ethics", "level": "basics",
     "title": "Copyright & Attribution",
     "learn": "AI-generated content and copyright law. Fair use, transformative work. Proper attribution of training data sources.",
     "try_code": "# Attribution checker: detect potential copyright issues",
     "try_demo": "cc_attribution"},
    {"id": "cc_bias_content", "book_id": "ethics", "level": "intermediate",
     "title": "Bias in Generated Content",
     "learn": "Stereotypes in AI text. Representation analysis. Debiasing strategies. Inclusive language guidelines.",
     "try_code": "# Bias scan: check for gendered language, stereotypes, exclusion",
     "try_demo": "cc_bias_scan"},
    {"id": "cc_deepfake", "book_id": "ethics", "level": "advanced",
     "title": "Deepfakes & Synthetic Media Ethics",
     "learn": "Detection of AI-generated content. Watermarking. Disclosure requirements. Consent in synthetic media.",
     "try_code": "# Detect: statistical patterns that distinguish AI vs human text",
     "try_demo": "cc_ai_detection"},
    {"id": "cc_responsible", "book_id": "ethics", "level": "expert",
     "title": "Responsible AI Content Frameworks",
     "learn": "Content moderation, safety guardrails, RLHF for alignment. Red-teaming AI systems. Content policy design.",
     "try_code": "# Safety pipeline: generate → filter → review → approve/reject",
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
