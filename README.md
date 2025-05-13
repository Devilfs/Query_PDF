# Query_PDF
# PDF ChatBot with Local LLM

A context-aware chatbot that answers questions based on PDF content using local AI models. Built for privacy-conscious users who need document analysis without cloud services.



## Technologies Used

### Core Components
- **LLM Inference**: `llama.cpp` with Phi-3-mini-4k-instruct (4-bit quantized)
- **Embeddings**: `all-mpnet-base-v2` Sentence Transformer model
- **Vector DB**: ChromaDB for efficient similarity search
- **PDF Processing**: PyMuPDF (fitz) for text extraction
- **Web Framework**: Flask with async support via Eventlet

### Key Libraries
- **NLP Processing**: Hugging Face Transformers
- **Vector Operations**: PyTorch (CPU/CUDA)
- **API Handling**: Flask-CORS for cross-origin support

## Features

- 📄 PDF document processing with smart text chunking
- � Context-aware answers using vector similarity search
- 🚀 Local AI inference with Phi-3 model
- ⚡ Async request handling for better performance
- 🔒 Full local execution - no data leaves your machine

## Installation

### Requirements
- Python 3.8+
- 8GB+ RAM (16GB recommended)
- NVIDIA GPU (optional but recommended for better performance)

1. **Clone Repository**
   ```bash
   git clone https://github.com/yourusername/pdf-chatbot.git
   cd pdf-chatbot
