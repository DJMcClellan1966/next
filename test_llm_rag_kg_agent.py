"""
Test LLM + RAG + Knowledge Graph Agent
"""
import sys
from pathlib import Path
import numpy as np
import warnings

warnings.filterwarnings('ignore')
sys.path.insert(0, str(Path(__file__).parent))

print("="*80)
print("LLM + RAG + KNOWLEDGE GRAPH AGENT TEST")
print("="*80)

try:
    from ml_toolbox import MLToolbox
    print("\n[OK] ML Toolbox imported")
except ImportError as e:
    print(f"\n[ERROR] Import failed: {e}")
    sys.exit(1)

# Initialize toolbox
toolbox = MLToolbox()

# Test LLM+RAG+KG Agent
if toolbox.llm_rag_kg_agent:
    print("\n[OK] LLM+RAG+KG Agent available")
    agent = toolbox.llm_rag_kg_agent
    
    # Test 1: Add Knowledge
    print("\n1. ADDING KNOWLEDGE")
    print("-"*80)
    agent.add_knowledge(
        "Random Forest is an ensemble method that uses multiple decision trees",
        doc_id="rf_info",
        add_to_kg=True,
        add_to_rag=True
    )
    agent.add_knowledge(
        "Decision trees are tree-based models that make decisions at each node",
        doc_id="dt_info",
        add_to_kg=True,
        add_to_rag=True
    )
    print("Added knowledge about Random Forest and Decision Trees")
    
    # Test 2: Process Query
    print("\n2. PROCESSING QUERY")
    print("-"*80)
    result = agent.process("Explain Random Forest")
    print(f"Query: {result['query']}")
    print(f"Response: {result['response']['text']}")
    print(f"Reasoning: {result['reasoning']}")
    print(f"Sources: {result['sources']}")
    
    # Test 3: Knowledge Graph Query
    print("\n3. KNOWLEDGE GRAPH QUERY")
    print("-"*80)
    if agent.kg_agent:
        kg = agent.kg_agent.get_graph()
        stats = kg.get_statistics()
        print(f"KG Statistics: {stats}")
        
        # Query graph
        kg_result = agent.kg_agent.query_graph("Find Random Forest")
        print(f"KG Query Result: {kg_result}")
    
    # Test 4: RAG Retrieval
    print("\n4. RAG RETRIEVAL")
    print("-"*80)
    if agent.rag_system:
        retrieved = agent.rag_system.retriever.retrieve("Random Forest", top_k=2)
        print(f"Retrieved {len(retrieved)} documents:")
        for doc in retrieved:
            print(f"  - {doc['content'][:100]}... (score: {doc['score']:.2f})")
    
    # Test 5: Statistics
    print("\n5. AGENT STATISTICS")
    print("-"*80)
    stats = agent.get_statistics()
    print(f"Components: {stats['components']}")
    if 'rag' in stats:
        print(f"RAG: {stats['rag']}")
    if 'knowledge_graph' in stats:
        print(f"Knowledge Graph: {stats['knowledge_graph']}")
    
    print("\n" + "="*80)
    print("LLM + RAG + KG AGENT TEST COMPLETE")
    print("="*80)
else:
    print("\n[SKIP] LLM+RAG+KG Agent not available")

# Test Agent Builder
print("\n" + "="*80)
print("AGENT BUILDER TEST")
print("="*80)

if toolbox.agent_builder:
    print("\n[OK] Agent Builder available")
    builder = toolbox.agent_builder
    
    # Test 1: Build ML Agent
    print("\n1. BUILDING ML AGENT")
    print("-"*80)
    ml_agent = builder.build_ml_agent(toolbox=toolbox)
    print(f"ML Agent built: {ml_agent is not None}")
    
    # Test 2: Build Custom Agent
    print("\n2. BUILDING CUSTOM AGENT")
    print("-"*80)
    custom_agent = (builder
        .set_name("CustomMLAgent")
        .set_description("Custom ML agent")
        .add_capability("classification")
        .add_capability("regression")
        .add_knowledge_domain("custom_ml", [
            "Custom knowledge 1",
            "Custom knowledge 2"
        ])
        .build(toolbox=toolbox))
    print(f"Custom Agent built: {custom_agent is not None}")
    
    # Test 3: Process with Custom Agent
    print("\n3. PROCESSING WITH CUSTOM AGENT")
    print("-"*80)
    result = custom_agent.process("How do I classify data?")
    print(f"Response: {result['response']['text']}")
    
    print("\n" + "="*80)
    print("AGENT BUILDER TEST COMPLETE")
    print("="*80)
else:
    print("\n[SKIP] Agent Builder not available")
