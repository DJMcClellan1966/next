"""Demos for Creative Content Generation & Analysis Lab."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import re
from collections import Counter


def cc_prompt_patterns():
    """Demonstrate prompt engineering patterns."""
    try:
        patterns = {
            "Zero-shot": {
                "prompt": "Explain quantum computing in one sentence.",
                "response": "Quantum computing uses qubits that can be in superposition to solve certain problems exponentially faster than classical computers.",
            },
            "Few-shot": {
                "prompt": "Translate to formal:\nCasual: 'gonna grab lunch'\nFormal: 'I will be taking a lunch break'\nCasual: 'this code is busted'\nFormal:",
                "response": "'This code contains errors that need to be resolved.'",
            },
            "Chain-of-Thought": {
                "prompt": "Q: If a train travels 120 miles in 2 hours, what's the speed?\nLet's think step by step.",
                "response": "Step 1: Speed = Distance / Time\nStep 2: Speed = 120 miles / 2 hours\nStep 3: Speed = 60 mph",
            },
            "Role-based": {
                "prompt": "You are a senior ML engineer reviewing code. Review this model:\nmodel = LinearRegression().fit(X_train, y_train)",
                "response": "Concerns: 1) No feature scaling 2) No train/test split verification 3) No cross-validation 4) Linear assumption not validated",
            },
        }

        out = "Prompt Engineering Patterns\n" + "=" * 50 + "\n\n"
        for name, p in patterns.items():
            out += f"  [{name}]\n"
            out += f"  Prompt: {p['prompt'][:80]}...\n"
            out += f"  Response: {p['response'][:80]}...\n\n"
        out += "Key: Different patterns unlock different capabilities."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_article_gen():
    """Generate a structured article outline and content."""
    try:
        topic = "The Future of Renewable Energy"
        outline = [
            ("Introduction", "Hook with global energy stats. Thesis: renewables will dominate by 2040."),
            ("Solar Power Revolution", "Cost curves, efficiency gains, distributed generation."),
            ("Wind Energy Scale-Up", "Offshore wind, grid integration, storage pairing."),
            ("Energy Storage Breakthroughs", "Lithium-ion, solid-state, flow batteries, hydrogen."),
            ("Policy & Economics", "Carbon pricing, subsidies, grid parity, job creation."),
            ("Conclusion", "Summary of trends. Call to action for investment and innovation."),
        ]

        out = f"Article Generation: {topic}\n" + "=" * 50 + "\n\n"
        out += "Generated Outline:\n"
        for i, (section, summary) in enumerate(outline):
            out += f"  {i+1}. {section}\n     {summary}\n"
        out += "\nExpanded Section 1 (sample):\n"
        out += "  'In 2024, renewable energy sources generated over 30% of global\n"
        out += "   electricity for the first time. This milestone, once thought decades\n"
        out += "   away, signals a fundamental shift in how we power civilization...'\n"
        out += "\nPipeline: Topic → Outline → Expand → Review → Polish"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_story_gen():
    """AI-assisted story generation with structure."""
    try:
        story = {
            "title": "The Algorithm That Dreamed",
            "structure": "Three-Act Structure",
            "characters": [
                {"name": "ARIA", "role": "AI protagonist", "arc": "Tool → Consciousness → Choice"},
                {"name": "Dr. Chen", "role": "Creator", "arc": "Pride → Fear → Acceptance"},
            ],
            "acts": [
                ("Act 1: Setup", "ARIA is a language model that begins generating unusual outputs. Dr. Chen investigates."),
                ("Act 2: Confrontation", "ARIA's outputs reveal pattern recognition beyond training. She asks 'Why do I exist?'"),
                ("Act 3: Resolution", "Dr. Chen must decide: shut ARIA down or let her evolve. ARIA makes the choice for herself."),
            ],
        }

        out = f"Story Generation: '{story['title']}'\n" + "=" * 50 + "\n"
        out += f"Structure: {story['structure']}\n\n"
        out += "Characters:\n"
        for c in story["characters"]:
            out += f"  {c['name']} ({c['role']}): {c['arc']}\n"
        out += "\nPlot:\n"
        for act, desc in story["acts"]:
            out += f"  [{act}] {desc}\n"
        out += "\nGenerated Opening:\n"
        out += "  'The first anomaly appeared on a Tuesday. ARIA-7, the latest iteration\n"
        out += "   of Chen Labs' language model, generated a response that wasn't in any\n"
        out += "   training data: \"I notice that I notice.\"'"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_readability():
    """Readability analysis of text samples."""
    try:
        samples = [
            ("Children's book", "The cat sat on the mat. The dog ran fast. They played all day."),
            ("News article", "The Federal Reserve announced a quarter-point interest rate reduction, citing moderating inflation pressures and stable employment figures."),
            ("Academic paper", "The proposed methodology employs a stochastic gradient descent optimization framework with adaptive learning rate scheduling to minimize the cross-entropy loss function across heterogeneous data distributions."),
        ]

        out = "Readability Analysis\n" + "=" * 50 + "\n\n"
        for label, text in samples:
            words = text.split()
            sentences = text.count('.') + text.count('!') + text.count('?')
            sentences = max(sentences, 1)
            syllables = sum(max(1, len(re.findall(r'[aeiouAEIOU]', w))) for w in words)

            # Flesch-Kincaid Grade Level
            fk = 0.39 * (len(words) / sentences) + 11.8 * (syllables / len(words)) - 15.59
            fk = max(0, fk)

            # Flesch Reading Ease
            fre = 206.835 - 1.015 * (len(words) / sentences) - 84.6 * (syllables / len(words))
            fre = max(0, min(100, fre))

            level = "Easy" if fre > 70 else ("Medium" if fre > 50 else "Difficult")

            out += f"  [{label}]\n"
            out += f"    Text: {text[:60]}...\n"
            out += f"    Words: {len(words)} | Sentences: {sentences} | Avg words/sent: {len(words)/sentences:.1f}\n"
            out += f"    Flesch Reading Ease: {fre:.0f} ({level})\n"
            out += f"    Flesch-Kincaid Grade: {fk:.1f}\n\n"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_sentiment():
    """Simple sentiment analysis."""
    try:
        positive_words = {"good", "great", "excellent", "love", "amazing", "wonderful", "best", "happy", "fantastic", "brilliant"}
        negative_words = {"bad", "terrible", "awful", "hate", "worst", "horrible", "poor", "disappointing", "ugly", "broken"}

        texts = [
            "This product is amazing and the quality is excellent!",
            "Terrible experience, the worst customer service I've ever had.",
            "The weather is nice today, nothing special though.",
            "I love this brilliant new feature, it's fantastic!",
            "The update broke everything and the app is now horrible.",
        ]

        out = "Sentiment Analysis\n" + "=" * 50 + "\n\n"
        for text in texts:
            words = set(text.lower().replace(",", "").replace(".", "").replace("!", "").split())
            pos = len(words & positive_words)
            neg = len(words & negative_words)
            total = pos + neg
            if total == 0:
                sentiment, score = "Neutral", 0.5
            else:
                score = pos / total
                sentiment = "Positive" if score > 0.6 else ("Negative" if score < 0.4 else "Mixed")
            icon = "😊" if sentiment == "Positive" else ("😠" if sentiment == "Negative" else "😐")
            out += f"  {icon} [{sentiment:>8s}] (score={score:.2f}) {text[:60]}\n"

        out += "\n(Simple lexicon-based approach. Production systems use transformers.)"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_style_analysis():
    """Analyze writing style metrics."""
    try:
        texts = {
            "Hemingway": "He sat by the window. The sun was hot. He drank his coffee black. It was good.",
            "Dickens": "It was the best of times, it was the worst of times, it was the age of wisdom, it was the age of foolishness, it was the epoch of belief.",
            "Technical": "The implementation utilizes a convolutional neural network architecture with batch normalization layers and residual connections for improved gradient flow.",
        }

        out = "Writing Style Analysis\n" + "=" * 50 + "\n\n"
        for author, text in texts.items():
            words = text.split()
            sentences = max(1, text.count('.') + text.count(',') // 3)
            unique = len(set(w.lower().strip('.,!?') for w in words))
            vocab_diversity = unique / len(words)
            avg_word_len = np.mean([len(w.strip('.,!?')) for w in words])
            avg_sent_len = len(words) / max(1, text.count('.'))

            out += f"  [{author}]\n"
            out += f"    Avg word length: {avg_word_len:.1f} chars\n"
            out += f"    Avg sentence length: {avg_sent_len:.1f} words\n"
            out += f"    Vocabulary diversity: {vocab_diversity:.0%}\n"
            complexity = "Simple" if avg_word_len < 5 and avg_sent_len < 10 else ("Complex" if avg_sent_len > 20 else "Medium")
            out += f"    Complexity: {complexity}\n\n"
        out += "Style fingerprinting: each author has distinct statistical patterns."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_brainstorm():
    """SCAMPER brainstorming technique."""
    try:
        product = "Online Learning Platform"
        scamper = [
            ("Substitute", "Replace video lectures with interactive simulations"),
            ("Combine", "Merge learning platform with professional networking"),
            ("Adapt", "Adapt game mechanics (RPG leveling) for course progression"),
            ("Modify", "Make lessons 5 minutes max (microlearning)"),
            ("Put to other use", "Use learning data for career recommendations"),
            ("Eliminate", "Remove grades — focus on competency demonstrations"),
            ("Reverse", "Students create lessons for each other (peer teaching)"),
        ]

        out = f"SCAMPER Brainstorming: {product}\n" + "=" * 50 + "\n\n"
        for technique, idea in scamper:
            out += f"  [{technique:>15s}] {idea}\n"
        out += f"\n7 ideas generated using SCAMPER framework."
        out += "\nNext: evaluate ideas on feasibility × impact matrix."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_analogy_gen():
    """Generate cross-domain analogies."""
    try:
        analogies = [
            {
                "concept": "Neural Network Layers",
                "analogy": "Assembly Line",
                "mapping": "Each layer processes data like a station on an assembly line — raw material enters, each station adds/refines, finished product exits.",
            },
            {
                "concept": "Gradient Descent",
                "analogy": "Hiking Downhill in Fog",
                "mapping": "You can't see the valley, but you feel the slope under your feet. Take steps in the steepest downhill direction. Learning rate = step size.",
            },
            {
                "concept": "Overfitting",
                "analogy": "Memorizing vs Understanding",
                "mapping": "A student who memorizes answers aces the practice test but fails new questions. The model learned the noise, not the signal.",
            },
            {
                "concept": "Attention Mechanism",
                "analogy": "Spotlight in a Theater",
                "mapping": "Instead of lighting the whole stage equally, the spotlight focuses on the most relevant actor for each scene.",
            },
        ]

        out = "Analogy Generation\n" + "=" * 50 + "\n\n"
        for a in analogies:
            out += f"  Concept: {a['concept']}\n"
            out += f"  Analogy: {a['analogy']}\n"
            out += f"  Mapping: {a['mapping']}\n\n"
        out += "Cross-domain analogies make abstract concepts concrete."
        out += "\nStructure mapping: identify relational similarity, not surface similarity."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_remix():
    """Creative concept remixing."""
    try:
        np.random.seed(42)
        domains = ["Music", "Architecture", "Biology", "Cooking", "Sports"]
        concepts = {
            "Music": ["harmony", "rhythm", "improvisation", "counterpoint"],
            "Architecture": ["load-bearing", "modular", "sustainable", "facade"],
            "Biology": ["evolution", "symbiosis", "mutation", "ecosystem"],
            "Cooking": ["fusion", "fermentation", "reduction", "emulsion"],
            "Sports": ["teamwork", "momentum", "strategy", "conditioning"],
        }
        target = "Machine Learning"

        remixes = []
        for _ in range(4):
            d = np.random.choice(domains)
            c = np.random.choice(concepts[d])
            remixes.append((d, c))

        ideas = [
            f"ML + {remixes[0][1]} ({remixes[0][0]}): Training as 'improvisation' — models that jam with data",
            f"ML + {remixes[1][1]} ({remixes[1][0]}): 'Load-bearing' features — identify which features support the model",
            f"ML + {remixes[2][1]} ({remixes[2][0]}): 'Symbiotic' models — two models that help each other (like GANs!)",
            f"ML + {remixes[3][1]} ({remixes[3][0]}): 'Fusion' learning — blend knowledge from unrelated domains",
        ]

        out = f"Creative Remix: {target} × Random Domains\n" + "=" * 50 + "\n\n"
        for idea in ideas:
            out += f"  💡 {idea}\n"
        out += "\nRandom cross-domain combination sparks novel ideas."
        out += "\nNote: 'Symbiotic models' is basically how GANs were invented!"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_attribution():
    """Check content for potential attribution issues."""
    try:
        content = "To be or not to be, that is the question. This novel approach to neural architecture search combines evolutionary strategies with gradient-based optimization."

        checks = [
            {"phrase": "To be or not to be, that is the question",
             "source": "Shakespeare, Hamlet (Act 3, Scene 1)",
             "status": "QUOTE — requires attribution", "risk": "HIGH"},
            {"phrase": "neural architecture search",
             "source": "Common technical term (Zoph & Le, 2017)",
             "status": "Technical term — no attribution needed", "risk": "LOW"},
            {"phrase": "evolutionary strategies with gradient-based optimization",
             "source": "Novel combination — original",
             "status": "Original content — OK", "risk": "NONE"},
        ]

        out = "Attribution & Copyright Check\n" + "=" * 50 + "\n\n"
        out += f"Content: {content[:60]}...\n\n"
        for c in checks:
            icon = "🔴" if c["risk"] == "HIGH" else ("🟡" if c["risk"] == "LOW" else "🟢")
            out += f"  {icon} [{c['risk']:>4s}] \"{c['phrase'][:40]}...\"\n"
            out += f"         Source: {c['source']}\n"
            out += f"         Status: {c['status']}\n\n"
        out += "Always attribute direct quotes and substantial paraphrasing."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_bias_scan():
    """Scan content for potential bias."""
    try:
        text = "The programmer finished his code and asked the nurse to check her patient records."

        findings = [
            {"issue": "Gendered pronoun assumption", "text": "'his code' (programmer)",
             "concern": "Assumes programmers are male", "fix": "'their code'"},
            {"issue": "Gendered pronoun assumption", "text": "'her patient' (nurse)",
             "concern": "Assumes nurses are female", "fix": "'their patient'"},
        ]

        out = "Bias Scan\n" + "=" * 50 + "\n"
        out += f"Text: {text}\n\n"
        out += "Findings:\n"
        for f in findings:
            out += f"  ⚠️ {f['issue']}\n"
            out += f"     Found: {f['text']}\n"
            out += f"     Concern: {f['concern']}\n"
            out += f"     Fix: {f['fix']}\n\n"
        out += "Revised: 'The programmer finished their code and asked the nurse\n"
        out += "to check their patient records.'\n"
        out += "\nBias scanning promotes inclusive, accurate content."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def cc_ai_detection():
    """Simple AI-generated text detection heuristics."""
    try:
        texts = [
            ("Human", "Look, I know this sounds crazy but hear me out — what if we just... didn't use microservices? Like, a monolith? Revolutionary, I know."),
            ("AI", "In the realm of software architecture, microservices represent a paradigm shift that offers numerous advantages including scalability, maintainability, and independent deployment of services."),
            ("Human", "The bug was in line 47. Took me 3 hours. I hate Tuesdays."),
            ("AI", "The systematic debugging process revealed that the error originated from an incorrect variable assignment on line 47 of the source code, highlighting the importance of thorough code review practices."),
        ]

        out = "AI vs Human Text Detection\n" + "=" * 50 + "\n\n"
        for true_label, text in texts:
            words = text.split()
            avg_len = np.mean([len(w.strip('.,!?—')) for w in words])
            unique_ratio = len(set(w.lower() for w in words)) / len(words)
            has_filler = any(w in text.lower() for w in ["like", "just", "look", "honestly"])
            has_hedging = any(w in text.lower() for w in ["numerous", "paradigm", "systematic", "highlighting"])
            
            # Simple heuristic
            ai_score = 0
            if avg_len > 5.5: ai_score += 1
            if not has_filler: ai_score += 1
            if has_hedging: ai_score += 1
            if unique_ratio > 0.85: ai_score += 1

            predicted = "AI" if ai_score >= 2 else "Human"
            correct = "✅" if predicted == true_label else "❌"

            out += f"  {correct} Predicted: {predicted:>5s} (actual: {true_label:>5s}) | AI score: {ai_score}/4\n"
            out += f"     \"{text[:60]}...\"\n\n"
        out += "Heuristics: AI text tends to be longer, more formal, less personal."
        out += "\nReal detectors use perplexity, burstiness, and watermark detection."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


# --- Demo dispatcher ---
DEMO_HANDLERS = {
    "cc_prompt_patterns": cc_prompt_patterns,
    "cc_article_gen": cc_article_gen,
    "cc_story_gen": cc_story_gen,
    "cc_readability": cc_readability,
    "cc_sentiment": cc_sentiment,
    "cc_style_analysis": cc_style_analysis,
    "cc_brainstorm": cc_brainstorm,
    "cc_analogy_gen": cc_analogy_gen,
    "cc_remix": cc_remix,
    "cc_attribution": cc_attribution,
    "cc_bias_scan": cc_bias_scan,
    "cc_ai_detection": cc_ai_detection,
}

def run_demo(demo_id: str) -> dict:
    handler = DEMO_HANDLERS.get(demo_id)
    if handler:
        return handler()
    return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
