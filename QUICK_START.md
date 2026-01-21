# Quick Start: Next Level Platform

Get started with the production-ready AI platform in minutes!

---

## 🚀 **Installation**

### **1. Install Dependencies**

```bash
# Install Python dependencies
pip install -r requirements.txt

# For GPU support (optional)
pip install faiss-gpu  # Instead of faiss-cpu
```

### **2. Verify Installation**

```bash
python -c "from vector_db import FAISSVectorDB; print('Vector DB ready!')"
python -c "from rag import RAGSystem; print('RAG System ready!')"
```

---

## 🎯 **Quick Start: RAG System**

### **1. Initialize Components**

```python
from quantum_kernel import get_kernel, KernelConfig
from llm.quantum_llm_standalone import StandaloneQuantumLLM
from vector_db import FAISSVectorDB
from rag import RAGSystem

# Configure kernel
config = KernelConfig(
    embedding_model='all-MiniLM-L6-v2',
    use_sentence_transformers=True,
    cache_type='lru'
)
kernel = get_kernel(config)

# Initialize LLM
llm = StandaloneQuantumLLM(kernel=kernel)

# Initialize vector database
vector_db = FAISSVectorDB(
    dimension=384,  # Sentence transformer dimension
    index_type='IP',  # Inner product for cosine similarity
    use_gpu=False  # Set to True if GPU available
)

# Initialize RAG system
rag = RAGSystem(kernel=kernel, llm=llm, vector_db=vector_db)
```

### **2. Add Knowledge Base**

```python
# Add documents
documents = [
    "Machine learning is a subset of artificial intelligence",
    "Neural networks are inspired by biological neurons",
    "Deep learning uses multiple layers of neural networks",
    # ... add more documents
]

metadata = [
    {"source": "intro.pdf", "page": 1},
    {"source": "intro.pdf", "page": 2},
    {"source": "advanced.pdf", "page": 10},
    # ... metadata for each document
]

rag.add_knowledge(documents, metadata)
```

### **3. Generate with RAG**

```python
# Query the system
result = rag.generate(
    query="What is machine learning?",
    top_k=5,
    max_length=200
)

print("Answer:", result['answer'])
print("\nSources:")
for citation in result['citations']:
    print(f"  {citation}")
```

---

## 🌐 **Quick Start: Production API**

### **1. Start API Server**

```bash
# Option 1: Direct
python -m api.server

# Option 2: Using uvicorn
uvicorn api.server:app --host 0.0.0.0 --port 8000 --reload

# Option 3: Docker
docker-compose up
```

### **2. Test API**

```bash
# Health check
curl http://localhost:8000/api/health

# Search
curl -X POST http://localhost:8000/api/search \
  -H "Content-Type: application/json" \
  -d '{"query": "machine learning", "top_k": 5}'

# Generate (RAG)
curl -X POST http://localhost:8000/api/generate \
  -H "Content-Type: application/json" \
  -d '{"prompt": "What is machine learning?", "top_k": 5}'

# Chat
curl -X POST http://localhost:8000/api/chat \
  -H "Content-Type: application/json" \
  -d '{"message": "Tell me about neural networks"}'
```

### **3. WebSocket Streaming**

```python
import asyncio
import websockets
import json

async def stream_chat():
    async with websockets.connect("ws://localhost:8000/api/stream") as websocket:
        await websocket.send(json.dumps({
            "query": "What is deep learning?",
            "top_k": 5
        }))
        
        while True:
            response = await websocket.recv()
            data = json.loads(response)
            
            print(data['token'], end='', flush=True)
            
            if data['done']:
                print("\n\nSources:", data.get('sources', []))
                break

asyncio.run(stream_chat())
```

---

## 📊 **Example: Complete Workflow**

```python
from quantum_kernel import get_kernel, KernelConfig
from llm.quantum_llm_standalone import StandaloneQuantumLLM
from vector_db import FAISSVectorDB
from rag import RAGSystem

# 1. Initialize
config = KernelConfig(use_sentence_transformers=True)
kernel = get_kernel(config)
llm = StandaloneQuantumLLM(kernel=kernel)
vector_db = FAISSVectorDB(dimension=384, index_type='IP')
rag = RAGSystem(kernel=kernel, llm=llm, vector_db=vector_db)

# 2. Add knowledge
documents = [
    "Python is a high-level programming language",
    "JavaScript is used for web development",
    "Machine learning involves training models on data"
]
rag.add_knowledge(documents)

# 3. Query
result = rag.generate("What is Python?", top_k=3)
print(result['answer'])
print("\nSources:")
for citation in result['citations']:
    print(f"  {citation}")

# 4. Save vector database
vector_db.save("data/knowledge_base.faiss")

# 5. Load later
vector_db.load("data/knowledge_base.faiss")
```

---

## 🐳 **Docker Deployment**

### **Build and Run**

```bash
# Build image
docker build -t quantum-ai-platform .

# Run container
docker run -p 8000:8000 -v $(pwd)/data:/app/data quantum-ai-platform

# Or use docker-compose
docker-compose up
```

---

## 📚 **API Documentation**

Once the server is running, visit:
- **Swagger UI:** http://localhost:8000/docs
- **ReDoc:** http://localhost:8000/redoc

---

## 🎯 **Next Steps**

1. **Add your documents** to the knowledge base
2. **Fine-tune** the LLM on your domain
3. **Deploy** to production (cloud/on-prem)
4. **Integrate** with your applications

---

**You now have a production-ready AI platform!** 🚀
