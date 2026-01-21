# Troubleshooting Guide

Common issues and solutions.

---

## 🔌 **Port Already in Use**

### **Error:**
```
ERROR: [Errno 10048] error while attempting to bind on address ('0.0.0.0', 8000): only one usage of each socket address (protocol/network address/port) is normally permitted
```

### **Solutions:**

#### **Option 1: Use Different Port (Recommended)**

```bash
# Set PORT environment variable
$env:PORT=8001
python -m api.server

# Or use the helper script
python start_server.py
```

The `start_server.py` script automatically finds a free port.

#### **Option 2: Kill Process Using Port 8000**

**Find the process:**
```bash
netstat -ano | findstr :8000
```

**Kill the process (replace PID with actual process ID):**
```bash
taskkill /PID <PID> /F
```

**Or kill all processes on port 8000:**
```powershell
# PowerShell
Get-NetTCPConnection -LocalPort 8000 | Select-Object -ExpandProperty OwningProcess | ForEach-Object { Stop-Process -Id $_ -Force }
```

---

## 🐳 **Docker Issues**

### **Error:**
```
unable to get image 'next-ai-platform': error during connect: Get "http://%2F%2F.%2Fpipe%2FdockerDesktopLinuxEngine/v1.51/images/next-ai-platform/json": open //./pipe/dockerDesktopLinuxEngine: The system cannot find the file specified.
```

### **Solution:**

Docker Desktop is not running. Either:

1. **Start Docker Desktop** and wait for it to fully start
2. **Run without Docker** (recommended for development):

```bash
# Install dependencies
pip install -r requirements.txt

# Start server directly
python start_server.py
# or
python -m api.server
```

---

## 📦 **Missing Dependencies**

### **Error:**
```
ModuleNotFoundError: No module named 'faiss'
```

### **Solution:**

```bash
# Install FAISS (CPU version)
pip install faiss-cpu

# Or GPU version (if you have GPU)
pip install faiss-gpu
```

### **Error:**
```
ModuleNotFoundError: No module named 'fastapi'
```

### **Solution:**

```bash
# Install all dependencies
pip install -r requirements.txt
```

---

## 🚀 **FAISS Installation Issues**

### **Windows Issues:**

FAISS can be tricky on Windows. Try:

```bash
# Option 1: Use conda (recommended for Windows)
conda install -c conda-forge faiss-cpu

# Option 2: Use pip with pre-built wheel
pip install faiss-cpu --no-cache-dir

# Option 3: Build from source (advanced)
# See: https://github.com/facebookresearch/faiss/blob/main/INSTALL.md
```

### **Alternative: Use In-Memory Fallback**

If FAISS doesn't work, the system will fall back to in-memory search:

```python
# System will automatically fall back if FAISS not available
# Still works, just slower
```

---

## 🔧 **API Server Won't Start**

### **Check Dependencies:**

```bash
# Verify all dependencies installed
pip list | findstr fastapi
pip list | findstr uvicorn
pip list | findstr faiss
```

### **Check Port:**

```bash
# Test if port is free
python -c "import socket; s=socket.socket(); s.bind(('0.0.0.0', 8000)); print('Port 8000 is free')"
```

### **Run with Verbose Logging:**

```bash
python -m api.server --log-level debug
```

---

## 🧠 **LLM/RAG Not Working**

### **Error:**
```
RAG system not available
```

### **Solution:**

Make sure you've initialized the RAG system:

```python
from quantum_kernel import get_kernel
from llm.quantum_llm_standalone import StandaloneQuantumLLM
from vector_db import FAISSVectorDB
from rag import RAGSystem

# Initialize all components
kernel = get_kernel()
llm = StandaloneQuantumLLM(kernel=kernel)
vector_db = FAISSVectorDB(dimension=384, index_type='IP')
rag = RAGSystem(kernel=kernel, llm=llm, vector_db=vector_db)

# Add knowledge
rag.add_knowledge(documents=["Doc 1", "Doc 2", ...])
```

---

## 💾 **Vector Database Issues**

### **Error:**
```
Vector dimension mismatch
```

### **Solution:**

Make sure embedding dimension matches:

```python
# Check kernel embedding dimension
kernel = get_kernel()
dimension = kernel.config.embedding_dim

# Or use Sentence Transformer default
if kernel.use_sentence_transformers:
    dimension = 384  # Sentence transformer default

# Create vector DB with matching dimension
vector_db = FAISSVectorDB(dimension=dimension, index_type='IP')
```

---

## 🔍 **Slow Search Performance**

### **Solutions:**

1. **Use GPU acceleration:**
```python
vector_db = FAISSVectorDB(dimension=384, index_type='IP', use_gpu=True)
```

2. **Ensure Sentence Transformers installed:**
```bash
pip install sentence-transformers
```

3. **Check cache is working:**
```python
stats = kernel.get_stats()
print(f"Cache hit rate: {stats.get('cache_hit_rate', 0):.2%}")
```

---

## 📊 **Memory Issues**

### **Error:**
```
MemoryError: Unable to allocate array
```

### **Solutions:**

1. **Use LRU cache:**
```python
config = KernelConfig(cache_type='lru', cache_size=5000)
kernel = get_kernel(config)
```

2. **Reduce batch size:**
```python
# When embedding large batches
embeddings = kernel.embed_batch(documents, batch_size=16)  # Smaller batches
```

3. **Process in chunks:**
```python
# Process documents in chunks
chunk_size = 1000
for i in range(0, len(documents), chunk_size):
    chunk = documents[i:i+chunk_size]
    rag.add_knowledge(chunk)
```

---

## 🌐 **CORS Issues**

If you get CORS errors in the browser:

```python
# In api/server.py, configure CORS properly:
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:3000", "http://localhost:5173"],  # Add your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
```

---

## 🐛 **General Debugging**

### **Enable Verbose Logging:**

```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

### **Check System Stats:**

```bash
# API endpoint
curl http://localhost:8001/api/stats

# Or in Python
from quantum_kernel import get_kernel
stats = get_kernel().get_stats()
print(stats)
```

### **Test Individual Components:**

```python
# Test kernel
from quantum_kernel import get_kernel
kernel = get_kernel()
embedding = kernel.embed("test")
print(f"Embedding shape: {embedding.shape}")

# Test vector DB
from vector_db import FAISSVectorDB
vector_db = FAISSVectorDB(dimension=384)
test_vectors = [[0.1] * 384]
vector_db.add_vectors(test_vectors, [1])
print("Vector DB working!")

# Test RAG
from rag import RAGSystem
# ... test RAG system
```

---

## 📞 **Still Having Issues?**

1. Check the logs for detailed error messages
2. Verify all dependencies are installed correctly
3. Ensure Python version is 3.9+
4. Try the example code in `QUICK_START.md`
5. Check that all required files are present

---

**Common Solutions Summary:**
- ✅ Port busy? Use `start_server.py` (auto-finds free port)
- ✅ Docker not running? Use `python start_server.py` instead
- ✅ Missing dependencies? Run `pip install -r requirements.txt`
- ✅ FAISS issues? Try conda or use fallback
- ✅ Memory issues? Use LRU cache, smaller batches

**Most issues can be resolved by using the helper script:**
```bash
python start_server.py
```
