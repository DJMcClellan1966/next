"""Demos for Research & Knowledge Discovery Lab."""
import sys
from pathlib import Path
REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

import numpy as np
import re
from collections import Counter


def rd_tfidf_search():
    """TF-IDF search over a mini research corpus."""
    try:
        corpus = [
            "Neural networks learn hierarchical representations from data",
            "Reinforcement learning optimizes cumulative reward through trial and error",
            "Knowledge graphs represent entities and relationships as triples",
            "Bayesian inference updates beliefs given new evidence using Bayes theorem",
            "Generative adversarial networks produce realistic synthetic data",
            "Transformer attention mechanism computes weighted context for each token",
            "Graph neural networks propagate information along edges in a graph",
            "Variational autoencoders learn latent representations via reparameterization",
        ]
        query = "learning representations from data"

        # Simple TF-IDF
        all_docs = corpus + [query]
        vocab = sorted(set(w for doc in all_docs for w in doc.lower().split()))
        word2idx = {w: i for i, w in enumerate(vocab)}
        n_docs = len(corpus)

        def tfidf_vec(doc):
            words = doc.lower().split()
            tf = Counter(words)
            vec = np.zeros(len(vocab))
            for w, count in tf.items():
                if w in word2idx:
                    df = sum(1 for d in corpus if w in d.lower())
                    idf = np.log((n_docs + 1) / (df + 1)) + 1
                    vec[word2idx[w]] = (count / len(words)) * idf
            norm = np.linalg.norm(vec)
            return vec / norm if norm > 0 else vec

        q_vec = tfidf_vec(query)
        scores = [(i, np.dot(tfidf_vec(doc), q_vec)) for i, doc in enumerate(corpus)]
        scores.sort(key=lambda x: -x[1])

        out = f"Query: '{query}'\n\nTop results by TF-IDF cosine similarity:\n"
        for rank, (idx, score) in enumerate(scores[:5], 1):
            out += f"  {rank}. [{score:.3f}] {corpus[idx]}\n"
        out += f"\nVocabulary size: {len(vocab)} terms across {n_docs} documents"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_embedding_search():
    """Dense embedding search using random projections (simulated)."""
    try:
        np.random.seed(42)
        docs = [
            "deep learning neural networks",
            "reinforcement learning rewards",
            "knowledge graph triples",
            "Bayesian probability inference",
            "transformer attention mechanism",
        ]
        query = "neural network learning"

        # Simulate embeddings with random projection
        vocab = sorted(set(w for d in docs + [query] for w in d.lower().split()))
        dim = 32
        proj = np.random.randn(len(vocab), dim) * 0.3
        word2idx = {w: i for i, w in enumerate(vocab)}

        def embed(text):
            words = text.lower().split()
            vecs = [proj[word2idx[w]] for w in words if w in word2idx]
            if not vecs:
                return np.zeros(dim)
            v = np.mean(vecs, axis=0)
            return v / (np.linalg.norm(v) + 1e-10)

        q_emb = embed(query)
        results = [(i, np.dot(embed(d), q_emb)) for i, d in enumerate(docs)]
        results.sort(key=lambda x: -x[1])

        out = f"Query: '{query}' (dim={dim} embedding)\n\nResults:\n"
        for rank, (idx, score) in enumerate(results, 1):
            out += f"  {rank}. [{score:.3f}] {docs[idx]}\n"
        out += "\n(Using random projections as simulated embeddings)"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_rag_pipeline():
    """Demonstrate a RAG pipeline (retrieval + generation template)."""
    try:
        knowledge_base = {
            "backprop": "Backpropagation computes gradients via the chain rule, propagating error from output to input layers.",
            "attention": "Attention mechanism computes weighted sum of values based on query-key compatibility scores.",
            "dropout": "Dropout randomly zeroes activations during training to prevent co-adaptation and improve generalization.",
            "batch_norm": "Batch normalization normalizes layer inputs to stabilize training and allow higher learning rates.",
        }
        query = "How does attention work in transformers?"

        # Retrieval: keyword match
        scores = {}
        for key, text in knowledge_base.items():
            overlap = len(set(query.lower().split()) & set(text.lower().split()))
            scores[key] = overlap
        best_key = max(scores, key=scores.get)
        context = knowledge_base[best_key]

        # Generation template
        prompt = f"""Based on the following context, answer the question.

Context: {context}

Question: {query}

Answer: The attention mechanism works by computing compatibility scores between queries and keys,
then using these scores as weights to produce a weighted sum of the values. This allows the model
to focus on the most relevant parts of the input for each output position."""

        out = "RAG Pipeline Demo\n"
        out += "=" * 50 + "\n"
        out += f"Query: {query}\n"
        out += f"Retrieved chunk: [{best_key}] {context}\n"
        out += f"\nGenerated answer (template):\n{prompt.split('Answer: ')[1]}\n"
        out += "\nPipeline: Query → Retrieve → Augment Prompt → Generate"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_build_kg():
    """Build a small knowledge graph from research statements."""
    try:
        statements = [
            ("Neural Networks", "are_type_of", "Machine Learning"),
            ("Transformers", "use", "Attention"),
            ("Attention", "computes", "Weighted Context"),
            ("GPT", "is_a", "Transformer"),
            ("BERT", "is_a", "Transformer"),
            ("Backpropagation", "trains", "Neural Networks"),
            ("Gradient Descent", "used_by", "Backpropagation"),
            ("GANs", "are_type_of", "Neural Networks"),
            ("GANs", "invented_by", "Goodfellow"),
        ]

        entities = set()
        for s, _, o in statements:
            entities.add(s)
            entities.add(o)

        # Find connections
        adj = {}
        for s, r, o in statements:
            adj.setdefault(s, []).append((r, o))
            adj.setdefault(o, []).append((f"inv_{r}", s))

        out = "Knowledge Graph\n" + "=" * 50 + "\n"
        out += f"Entities: {len(entities)} | Triples: {len(statements)}\n\n"
        out += "Triples:\n"
        for s, r, o in statements:
            out += f"  ({s}) --[{r}]--> ({o})\n"
        out += f"\nEntity with most connections: "
        most = max(adj, key=lambda k: len(adj[k]))
        out += f"{most} ({len(adj[most])} connections)\n"
        out += f"\nPath: GPT → Transformer → Attention → Weighted Context"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_entity_extract():
    """Extract entities from research text using simple pattern matching."""
    try:
        text = """
        Vaswani et al. introduced the Transformer architecture in 2017, which uses
        multi-head self-attention. This led to models like BERT (Devlin et al., 2018)
        and GPT-3 (Brown et al., 2020). The key innovation was replacing recurrence
        with attention, enabling parallelization and better long-range dependencies.
        """

        # Simple NER patterns
        year_pattern = r'\b(19|20)\d{2}\b'
        model_names = ["Transformer", "BERT", "GPT-3"]
        author_pattern = r'([A-Z][a-z]+)\s+et al\.'

        years = re.findall(year_pattern, text)
        authors = re.findall(author_pattern, text)
        models = [m for m in model_names if m in text]

        out = "Entity Extraction\n" + "=" * 50 + "\n"
        out += f"Text: {text.strip()[:100]}...\n\n"
        out += f"Models found: {', '.join(models)}\n"
        out += f"Authors found: {', '.join(authors)}\n"
        out += f"Years found: {', '.join(set('20' + y if y == '17' or y == '18' or y == '20' else y for y in years))}\n"
        out += "\nExtracted triples:\n"
        out += "  (Vaswani, introduced, Transformer) [2017]\n"
        out += "  (Devlin, introduced, BERT) [2018]\n"
        out += "  (Brown, introduced, GPT-3) [2020]\n"
        out += "  (Transformer, uses, multi-head self-attention)\n"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_discover_connections():
    """Discover hidden connections via structural graph analysis."""
    try:
        np.random.seed(42)
        # Concept graph
        concepts = ["Attention", "Convolution", "Fourier", "Kernel", "Graph",
                     "Diffusion", "Markov", "Energy", "Gradient", "Entropy"]
        n = len(concepts)

        # Simulated adjacency (sparse connections)
        adj = np.zeros((n, n))
        edges = [(0,1), (0,3), (1,2), (2,3), (3,4), (4,5), (5,6), (6,7), (7,8), (8,9),
                 (0,9), (2,7), (1,5)]  # Hidden: Attention↔Entropy, Fourier↔Energy
        for i, j in edges:
            adj[i][j] = adj[j][i] = 1

        # Compute structural similarity (common neighbors)
        discoveries = []
        for i in range(n):
            for j in range(i + 1, n):
                if adj[i][j] == 0:
                    common = np.sum(adj[i] * adj[j])
                    if common >= 2:
                        discoveries.append((concepts[i], concepts[j], int(common)))

        discoveries.sort(key=lambda x: -x[2])

        out = "Hidden Connection Discovery\n" + "=" * 50 + "\n"
        out += f"Concept graph: {n} nodes, {len(edges)} edges\n\n"
        out += "Discovered implicit connections (≥2 common neighbors):\n"
        for a, b, common in discoveries[:5]:
            out += f"  {a} ↔ {b} (shared neighbors: {common})\n"
        out += "\nThese pairs are NOT directly connected but share structural context."
        out += "\nThis is analogous to link prediction in knowledge graphs."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_single_agent():
    """Demonstrate a single research agent with tool use."""
    try:
        tools = {
            "search": lambda q: f"Found 3 papers on '{q}'",
            "summarize": lambda t: f"Summary: {t[:50]}...",
            "calculate": lambda e: f"Result: {eval(e)}",
        }

        query = "What is the relationship between attention and convolution?"
        agent_log = []

        # Simulated ReAct loop
        agent_log.append(("Think", "I need to search for papers on attention and convolution"))
        agent_log.append(("Act", "search('attention convolution relationship')"))
        agent_log.append(("Observe", tools["search"]("attention convolution")))
        agent_log.append(("Think", "Found papers. Let me check if convolution can be expressed as attention"))
        agent_log.append(("Act", "search('convolution as special case of attention')"))
        agent_log.append(("Observe", tools["search"]("convolution special case attention")))
        agent_log.append(("Think", "Yes — local attention with fixed window = convolution. Conclusion ready."))
        agent_log.append(("Answer", "Convolution is a special case of attention with fixed local receptive field."))

        out = "Research Agent (ReAct Pattern)\n" + "=" * 50 + "\n"
        out += f"Query: {query}\n\n"
        for step, (action, detail) in enumerate(agent_log, 1):
            out += f"  Step {step} [{action:>8s}]: {detail}\n"
        out += "\nAgent completed in 4 reasoning steps with 2 tool calls."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_multi_agent():
    """Multi-agent research coordination demo."""
    try:
        agents = {
            "Literature": "I found 5 papers on graph neural networks for drug discovery.",
            "Methods": "GNNs encode molecular graphs; message passing aggregates neighbor features.",
            "Statistics": "Best reported accuracy: 89.3% (±2.1%) on MoleculeNet benchmark.",
            "Ethics": "Drug discovery AI must account for fairness across demographics.",
        }

        out = "Multi-Agent Research Coordination\n" + "=" * 50 + "\n"
        out += "Query: 'Graph neural networks for drug discovery'\n\n"
        out += "Orchestrator dispatches to specialist agents:\n\n"
        for name, response in agents.items():
            out += f"  [{name:>12s} Agent]: {response}\n"
        out += "\nOrchestrator synthesis:\n"
        out += "  GNNs are effective for drug discovery (89.3% accuracy),\n"
        out += "  encoding molecules as graphs with message passing.\n"
        out += "  5 key papers identified. Ethics review flags fairness concern."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_socratic():
    """Socratic research refinement demo."""
    try:
        dialogue = [
            ("User", "I want to study transformer efficiency"),
            ("Socratic Agent", "Interesting! When you say 'efficiency', do you mean computational cost, memory, or parameter count?"),
            ("User", "Computational cost — specifically the O(n²) attention"),
            ("Socratic Agent", "Good distinction. Are you interested in (a) approximating full attention, or (b) alternative architectures that avoid it?"),
            ("User", "Approximating full attention"),
            ("Socratic Agent", "Then your research question is: 'What methods approximate O(n²) self-attention with subquadratic complexity while preserving quality?' Shall I search for linear attention, sparse attention, or kernel-based methods?"),
            ("User", "All three please"),
            ("Socratic Agent", "Refined query: 'Subquadratic attention approximations: linear (Katharopoulos 2020), sparse (Zaheer 2020), and kernel-based (Choromanski 2020)' — searching..."),
        ]

        out = "Socratic Research Refinement\n" + "=" * 50 + "\n\n"
        for speaker, text in dialogue:
            prefix = "  Q:" if speaker == "User" else "  A:"
            out += f"{prefix} [{speaker}] {text}\n"
        out += "\nRefinement: vague topic → precise research question in 4 turns"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_trend_analysis():
    """Research trend analysis with keyword velocity."""
    try:
        np.random.seed(42)
        keywords = {
            "transformer": [10, 25, 60, 120, 200, 350, 500],
            "GAN": [5, 30, 80, 150, 180, 170, 160],
            "RL": [20, 25, 35, 45, 55, 60, 65],
            "diffusion": [1, 2, 5, 15, 80, 250, 400],
            "neuroevolution": [5, 8, 10, 12, 11, 10, 9],
        }
        years = list(range(2018, 2025))

        out = "Research Trend Analysis\n" + "=" * 50 + "\n"
        out += "Papers per year by keyword:\n\n"
        out += f"  {'Keyword':>16s}", 
        for y in years:
            out += f" {y}"
        out += "  Trend\n"
        out += "  " + "-" * 80 + "\n"

        for kw, counts in keywords.items():
            velocity = (counts[-1] - counts[-2]) / max(counts[-2], 1)
            trend = "🚀 Rising" if velocity > 0.3 else ("📉 Falling" if velocity < -0.1 else "→ Stable")
            out += f"  {kw:>16s}"
            for c in counts:
                out += f" {c:>4d}"
            out += f"  {trend}\n"

        out += "\n🚀 Emerging: transformer, diffusion (highest velocity)"
        out += "\n📉 Declining: GAN (saturating/declining)"
        out += "\n→ Stable: RL, neuroevolution"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_forecast():
    """S-curve technology adoption forecast."""
    try:
        # S-curve: adoption(t) = K / (1 + exp(-r*(t - t0)))
        K = 1000  # Carrying capacity
        r = 1.2   # Growth rate
        t0 = 2022 # Inflection point

        years = np.arange(2018, 2028)
        adoption = K / (1 + np.exp(-r * (years - t0)))

        out = "Technology Adoption Forecast (S-Curve)\n" + "=" * 50 + "\n"
        out += f"Model: K={K}, growth_rate={r}, inflection={t0}\n\n"
        for y, a in zip(years, adoption):
            bar = "█" * int(a / 20)
            marker = " ← inflection" if y == t0 else ""
            out += f"  {y}: {a:>7.0f} papers {bar}{marker}\n"
        out += f"\nPrediction: By 2027, ~{adoption[-1]:.0f} papers (approaching saturation at {K})"
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


def rd_ethics_check():
    """Ethical research review screening."""
    try:
        proposal = {
            "title": "Facial Recognition for Classroom Attention Monitoring",
            "methods": "CNN-based face detection, emotion classification, attention scoring",
            "data": "Student video recordings from university lectures",
        }
        checks = [
            ("Privacy", "HIGH", "Video recording of students raises significant privacy concerns"),
            ("Consent", "HIGH", "Informed consent required from all participants"),
            ("Bias", "MEDIUM", "Facial recognition has known demographic bias (Buolamwini 2018)"),
            ("Dual Use", "MEDIUM", "Surveillance technology could be misused for control"),
            ("Beneficence", "LOW", "Potential educational benefit if done ethically"),
        ]

        out = "Ethical Research Review\n" + "=" * 50 + "\n"
        out += f"Proposal: {proposal['title']}\n\n"
        out += "Ethical Screening Results:\n"
        for category, risk, note in checks:
            icon = "🔴" if risk == "HIGH" else ("🟡" if risk == "MEDIUM" else "🟢")
            out += f"  {icon} [{risk:>6s}] {category}: {note}\n"
        out += "\nRecommendation: REVISE — address HIGH-risk privacy and consent issues"
        out += "\nbefore proceeding. Consider anonymization and opt-in protocols."
        return {"ok": True, "output": out}
    except Exception as e:
        return {"ok": False, "output": "", "error": str(e)}


# --- Demo dispatcher ---
DEMO_HANDLERS = {
    "rd_tfidf_search": rd_tfidf_search,
    "rd_embedding_search": rd_embedding_search,
    "rd_rag_pipeline": rd_rag_pipeline,
    "rd_build_kg": rd_build_kg,
    "rd_entity_extract": rd_entity_extract,
    "rd_discover_connections": rd_discover_connections,
    "rd_single_agent": rd_single_agent,
    "rd_multi_agent": rd_multi_agent,
    "rd_socratic": rd_socratic,
    "rd_trend_analysis": rd_trend_analysis,
    "rd_forecast": rd_forecast,
    "rd_ethics_check": rd_ethics_check,
}

def run_demo(demo_id: str) -> dict:
    handler = DEMO_HANDLERS.get(demo_id)
    if handler:
        return handler()
    return {"ok": False, "output": "", "error": f"Unknown demo: {demo_id}"}
