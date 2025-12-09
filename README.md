# 📘 AI Syllabus Tutor - RAG & Gen AI Project

An AI-powered chatbot system that uses Retrieval-Augmented Generation (RAG) to answer questions based on syllabus material. The project consists of a FastAPI backend that handles RAG operations with FAISS vector search, and a Streamlit frontend for user interaction. The system uses local LLM models via LM Studio for generating answers.

---

## 🏗️ Project Architecture

```
AI-Tutor-Project/
├── backend/                    # FastAPI RAG service
│   ├── rag_service.py         # Main RAG service with FAISS and LLM integration
│   ├── main.py                # FastAPI application entry point
│   ├── requirements.txt       # Backend dependencies
│   ├── ml_book.index          # FAISS vector index (generated)
│   └── ml_book_chunks.json    # Document chunks (generated)
├── frontend/                   # Streamlit web interface
│   ├── app.py                 # Streamlit application
│   └── requirements.txt       # Frontend dependencies
├── syllabus_data/              # Source PDF documents
│   ├── MATHS/
│   └── ML/
├── LLMRAG_testing.ipynb       # Jupyter notebook for testing/documentation
└── README.md                  # This file
```

---

## 🚀 Features

- **📚 RAG (Retrieval-Augmented Generation)**: Semantic search using FAISS vector database
- **🤖 Local LLM Integration**: Uses LM Studio for local model inference
- **💬 Interactive Chat Interface**: Clean Streamlit UI with styled components
- **🔍 Context-Aware Answers**: Retrieves relevant chunks before generating responses
- **⚡ Fast Semantic Search**: FAISS-based similarity search for efficient retrieval
- **🛡️ Error Handling**: Comprehensive error handling for context length overflow and connection issues
- **📊 Customizable Parameters**: Adjustable top_k for context chunk retrieval

---

## 📋 Technology Stack

### Backend
- **FastAPI**: Modern Python web framework for building APIs
- **FAISS**: Facebook AI Similarity Search for vector similarity search
- **Sentence Transformers**: For generating text embeddings (`all-MiniLM-L6-v2`)
- **PyMuPDF (fitz)**: PDF text extraction
- **Uvicorn**: ASGI server for FastAPI
- **LM Studio**: Local LLM inference server

### Frontend
- **Streamlit**: Rapid web app development framework
- **Requests**: HTTP client for backend communication

---

## 🔧 Prerequisites

1. **Python 3.10+**
2. **LM Studio** installed and running with a model loaded
3. **Virtual Environment** (recommended)

---

## 📦 Installation & Setup

### 1. Clone the Repository
```bash
git clone <your-repo-url>
cd AI-Tutor-Project
```

### 2. Create and Activate Virtual Environment
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

### 3. Install Backend Dependencies
```bash
cd backend
pip install -r requirements.txt
```

### 4. Install Frontend Dependencies
```bash
cd ../frontend
pip install -r requirements.txt
```

### 5. Generate Vector Index (If Not Already Generated)

If you need to create the FAISS index from PDF files:

```python
# Use the script in main.py or LLMRAG_testing.ipynb
# Example workflow:
from main import extract_text_from_pdf, chunk_text
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np
import json

# Extract text from PDF
text = extract_text_from_pdf("path/to/pdf")

# Chunk the text
chunks = chunk_text(text, max_words=500)

# Generate embeddings
model = SentenceTransformer('all-MiniLM-L6-v2')
embeddings = model.encode(chunks)

# Create FAISS index
emb_matrix = np.array(embeddings).astype('float32')
index = faiss.IndexFlatL2(emb_matrix.shape[1])
index.add(emb_matrix)

# Save index and chunks
faiss.write_index(index, "ml_book.index")
with open("ml_book_chunks.json", "w") as f:
    json.dump(chunks, f)
```

---

## 🎯 Running the Application

### Step 1: Start LM Studio
1. Open LM Studio
2. Load your desired model (e.g., `gemma-3-12b-it`)
3. Start the local server (default: `http://localhost:1234`)

### Step 2: Start the Backend Server
```bash
cd backend
uvicorn rag_service:app --reload --port 8000
```

The API will be available at `http://localhost:8000`

**API Endpoints:**
- `POST /ask` - Submit a question and get an answer

**Example Request:**
```json
{
  "question": "What is machine learning?",
  "top_k": 3
}
```

**Example Response:**
```json
{
  "answer": "Machine learning is a subset of artificial intelligence...",
  "retrieved_count": 3
}
```

### Step 3: Start the Frontend
```bash
cd frontend
streamlit run app.py
```

The frontend will be available at `http://localhost:8501`

---

## 🔍 Configuration

### Backend Configuration (`backend/rag_service.py`)

```python
LMSTUDIO_URL = "http://localhost:1234/v1/chat/completions"
LM_MODEL_NAME = "gemma-3-12b-it"  # Change to your LM Studio model name
FAISS_INDEX_PATH = "ml_book.index"
CHUNKS_JSON_PATH = "ml_book_chunks.json"
TOP_K = 3                          # Default number of chunks to retrieve
MAX_CONTEXT_CHARS = 15000          # Maximum context length
REQUEST_TIMEOUT = 60               # Request timeout in seconds
```

### Frontend Configuration (`frontend/app.py`)

```python
RAG_BACKEND = os.getenv("RAG_BACKEND_URL", "http://localhost:8000/ask")
TIMEOUT = 60
```

You can override the backend URL using environment variables:
```bash
export RAG_BACKEND_URL="http://localhost:8000/ask"
```

---

## 📖 Usage

### Using the Web Interface

1. **Enter Your Question**: Type your question in the search box
2. **Adjust Context Chunks**: Use the slider to set how many context chunks to retrieve (1-5)
3. **Submit**: Click "🔍 Ask Question" button
4. **View Answer**: The answer will be displayed along with metadata about chunks used

### Using the API Directly

```python
import requests

response = requests.post(
    "http://localhost:8000/ask",
    json={
        "question": "What is machine learning?",
        "top_k": 3
    }
)

result = response.json()
print(result["answer"])
```

---

## 🛠️ How It Works

### RAG Pipeline

1. **Question Input**: User submits a question through the frontend
2. **Embedding Generation**: The question is converted to an embedding using Sentence Transformers
3. **Similarity Search**: FAISS searches for the most similar chunks in the vector database
4. **Context Retrieval**: Top-k most relevant chunks are retrieved
5. **Prompt Construction**: The question and retrieved chunks are combined into a prompt
6. **LLM Generation**: LM Studio generates an answer based on the context
7. **Response**: The answer is returned to the user

### Error Handling

The system includes comprehensive error handling for:
- **Context Length Overflow**: Detects when input exceeds model's context window
- **Connection Errors**: Handles backend/frontend communication issues
- **LM Studio Errors**: Specific handling for LLM-related errors

---

## 🐛 Troubleshooting

### Common Issues

#### 1. Context Length Overflow Error
**Error**: `"Context length overflow: The input is too long..."`

**Solutions**:
- Reduce the `top_k` value (try 1 or 2)
- Reload the model in LM Studio with a larger context length
- Ask more specific questions that require less context

#### 2. LM Studio Connection Error
**Error**: `"Failed to reach backend"`

**Solutions**:
- Ensure LM Studio is running and the server is started
- Verify the model is loaded in LM Studio
- Check that `LMSTUDIO_URL` matches your LM Studio server URL

#### 3. FAISS Index Not Found
**Error**: `"Failed to load index/chunks"`

**Solutions**:
- Generate the index using the workflow described in Installation
- Ensure `ml_book.index` and `ml_book_chunks.json` are in the `backend/` directory
- Check file paths in `rag_service.py`

#### 4. Import Errors
**Error**: Module not found

**Solutions**:
- Ensure virtual environment is activated
- Install all dependencies: `pip install -r requirements.txt`
- Verify Python version is 3.10+

---

## 📝 File Structure Details

### Backend Files

- **`rag_service.py`**: Main RAG service implementation
  - Loads FAISS index and chunks
  - Handles question processing and retrieval
  - Manages LLM communication via LM Studio
  - Error handling and logging

- **`main.py`**: PDF processing utilities
  - PDF text extraction
  - Text chunking functions
  - Embedding generation and FAISS index creation

### Frontend Files

- **`app.py`**: Streamlit web application
  - User interface components
  - Form handling and input validation
  - API communication with backend
  - Styled components with custom CSS

---

## 🔐 Environment Variables

```bash
# Optional: Override backend URL
export RAG_BACKEND_URL="http://localhost:8000/ask"
```

---

## 📊 Performance Considerations

- **Embedding Model**: `all-MiniLM-L6-v2` is fast and efficient for semantic search
- **FAISS Index**: In-memory search provides fast similarity matching
- **Context Limiting**: `MAX_CONTEXT_CHARS` prevents context overflow
- **Chunk Size**: Optimize chunk size (default 500 words) based on your documents

---

## 🔮 Future Enhancements

- [ ] Support for multiple PDF uploads through the UI
- [ ] Multiple subject/semester support
- [ ] Conversation history and context
- [ ] Export conversation logs
- [ ] Support for other document formats (PPT, DOCX)
- [ ] User authentication and multi-user support
- [ ] Advanced chunking strategies (semantic chunking)
- [ ] Model selection UI
- [ ] Analytics and usage statistics

---

## 📄 License

This project is open source and available for educational purposes.

---

## 👥 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

---

## 🙏 Acknowledgments

- **Sentence Transformers**: For embedding models
- **FAISS**: For efficient similarity search
- **FastAPI**: For the robust backend framework
- **Streamlit**: For rapid frontend development
- **LM Studio**: For local LLM inference



---

**Happy Learning! 🎓**
