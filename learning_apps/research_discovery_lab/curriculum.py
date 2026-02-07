"""
Curriculum: Intelligent Research & Knowledge Discovery Platform.
Semantic search, multi-agent research, knowledge graphs, Socratic refinement,
trend forecasting, and ethical review.
"""
from typing import Dict, Any, List

LEVELS = ["basics", "intermediate", "advanced", "expert"]

BOOKS = [
    {"id": "semantic", "name": "Semantic Search & Retrieval", "short": "Semantic", "color": "#2563eb"},
    {"id": "knowledge", "name": "Knowledge Graphs & Discovery", "short": "KG", "color": "#059669"},
    {"id": "agents", "name": "Multi-Agent Research Systems", "short": "Agents", "color": "#7c3aed"},
    {"id": "forecasting", "name": "Trend Forecasting & Ethics", "short": "Forecast", "color": "#dc2626"},
]

CURRICULUM: List[Dict[str, Any]] = [
    # --- Semantic Search & Retrieval ---
    {"id": "rd_tfidf", "book_id": "semantic", "level": "basics",
     "title": "TF-IDF & Bag of Words",
     "learn": "Term frequency × inverse document frequency. Sparse vector search over document corpora. Foundation of information retrieval.",
     "try_code": "from sklearn.feature_extraction.text import TfidfVectorizer\nvec = TfidfVectorizer(); X = vec.fit_transform(['doc1 text', 'doc2 text'])",
     "try_demo": "rd_tfidf_search"},
    {"id": "rd_embeddings", "book_id": "semantic", "level": "intermediate",
     "title": "Dense Embeddings & Vector Search",
     "learn": "Word2Vec, sentence transformers, embedding spaces. Cosine similarity for semantic retrieval. ANN indexes (FAISS, Annoy).",
     "try_code": "import numpy as np\n# Cosine similarity between embedding vectors\ndef cosine_sim(a, b): return np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b))",
     "try_demo": "rd_embedding_search"},
    {"id": "rd_rag", "book_id": "semantic", "level": "advanced",
     "title": "Retrieval-Augmented Generation (RAG)",
     "learn": "Retrieve relevant chunks → embed in LLM prompt → generate grounded answer. Chunking strategies, re-ranking, hybrid search.",
     "try_code": "# RAG pipeline: chunk → embed → index → query → retrieve → generate",
     "try_demo": "rd_rag_pipeline"},
    {"id": "rd_reranking", "book_id": "semantic", "level": "expert",
     "title": "Cross-Encoder Re-Ranking & Query Expansion",
     "learn": "Two-stage retrieval: fast recall (bi-encoder) → precise re-ranking (cross-encoder). HyDE, query expansion, pseudo-relevance feedback.",
     "try_code": "# Re-rank candidates by cross-encoder score",
     "try_demo": None},

    # --- Knowledge Graphs & Discovery ---
    {"id": "rd_kg_basics", "book_id": "knowledge", "level": "basics",
     "title": "Knowledge Graph Fundamentals",
     "learn": "Entities, relations, triples (subject, predicate, object). RDF, property graphs. Schema vs schema-free KGs.",
     "try_code": "# Knowledge triple: ('Einstein', 'developed', 'General Relativity')\nkg = [('Einstein', 'developed', 'Relativity'), ('Relativity', 'uses', 'Tensors')]",
     "try_demo": "rd_build_kg"},
    {"id": "rd_entity_linking", "book_id": "knowledge", "level": "intermediate",
     "title": "Entity Extraction & Linking",
     "learn": "Named entity recognition (NER) → entity disambiguation → link to KG nodes. SpaCy, Flair, or LLM-based extraction.",
     "try_code": "# Extract entities from text, link to knowledge base",
     "try_demo": "rd_entity_extract"},
    {"id": "rd_kg_reasoning", "book_id": "knowledge", "level": "advanced",
     "title": "Graph Reasoning & Link Prediction",
     "learn": "TransE, RotatE, ComplEx for embedding KGs. Link prediction: infer missing triples. Path-based reasoning.",
     "try_code": "# TransE: h + r ≈ t  (head + relation ≈ tail)",
     "try_demo": None},
    {"id": "rd_discovery", "book_id": "knowledge", "level": "expert",
     "title": "Automated Knowledge Discovery",
     "learn": "Mining implicit relationships. Hypothesis generation from KG structure. Cross-domain analogy detection via structural similarity.",
     "try_code": "# Discover hidden connections between research areas",
     "try_demo": "rd_discover_connections"},

    # --- Multi-Agent Research Systems ---
    {"id": "rd_agent_basics", "book_id": "agents", "level": "basics",
     "title": "Research Agent Architecture",
     "learn": "Single agent: perception → reasoning → action. Tool use (search, calculate, summarize). ReAct pattern: Reason + Act.",
     "try_code": "# Agent loop: observe → think → act → observe",
     "try_demo": "rd_single_agent"},
    {"id": "rd_multi_agent", "book_id": "agents", "level": "intermediate",
     "title": "Multi-Agent Coordination",
     "learn": "Specialist agents (literature, methods, stats). Orchestrator pattern. Shared memory/blackboard. Debate for verification.",
     "try_code": "# Orchestrator assigns subtasks to specialist agents",
     "try_demo": "rd_multi_agent"},
    {"id": "rd_socratic", "book_id": "agents", "level": "advanced",
     "title": "Socratic Research Refinement",
     "learn": "Agent asks clarifying questions to refine research direction. Iterative hypothesis refinement. Contradiction detection.",
     "try_code": "# Socratic loop: hypothesis → question → refine → repeat",
     "try_demo": "rd_socratic"},
    {"id": "rd_collab", "book_id": "agents", "level": "expert",
     "title": "Agent Collaboration & Consensus",
     "learn": "Voting mechanisms, weighted expert opinions, Delphi method with AI agents. Conflict resolution protocols.",
     "try_code": "# Agents debate, vote, and reach consensus on findings",
     "try_demo": None},

    # --- Trend Forecasting & Ethics ---
    {"id": "rd_trend", "book_id": "forecasting", "level": "basics",
     "title": "Research Trend Analysis",
     "learn": "Citation velocity, keyword frequency over time, co-authorship networks. Identifying emerging topics from publication data.",
     "try_code": "# Track keyword frequency over time to detect emerging trends",
     "try_demo": "rd_trend_analysis"},
    {"id": "rd_forecast", "book_id": "forecasting", "level": "intermediate",
     "title": "Precognitive Forecasting",
     "learn": "Time series of publication rates → predict breakout topics. Diffusion models for idea propagation. S-curve technology adoption.",
     "try_code": "# S-curve: adoption(t) = K / (1 + exp(-r*(t - t0)))",
     "try_demo": "rd_forecast"},
    {"id": "rd_ethics_review", "book_id": "forecasting", "level": "advanced",
     "title": "Ethical Research Review",
     "learn": "Automated ethics screening: bias detection, dual-use analysis, fairness audits. Moral reasoning frameworks applied to research.",
     "try_code": "# Screen research proposal for ethical concerns",
     "try_demo": "rd_ethics_check"},
    {"id": "rd_impact", "book_id": "forecasting", "level": "expert",
     "title": "Research Impact Prediction",
     "learn": "Predict citation count, h-index impact, societal influence. Feature engineering from paper metadata. Causal impact analysis.",
     "try_code": "# Predict paper impact from abstract, venue, author features",
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
