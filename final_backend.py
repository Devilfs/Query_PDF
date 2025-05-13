# eventlet MUST BE FIRST and use limited patching
import eventlet
eventlet.monkey_patch(select=True, socket=True, thread=True, time=True)
import torch
import sys
sys.setrecursionlimit(10000)  # Fix recursion depth issues

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import fitz  # PyMuPDF
import chromadb
from chromadb.utils import embedding_functions
from sentence_transformers import SentenceTransformer, models
from transformers import AutoTokenizer
import re
import os
import hashlib
import logging
from llama_cpp import Llama

# Suppress warnings
logging.getLogger("chromadb").setLevel(logging.WARNING)
logging.getLogger("urllib3").setLevel(logging.WARNING)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger(__name__)

app = Flask(__name__)
CORS(app)

# Configuration
MAX_CONTEXT_LENGTH = 6000  # Reduced to match model capacity
MAX_CHUNK_SIZE = 800       # Increased chunk size
MAX_FILE_SIZE = 50 * 1024 * 1024  # 50MB
CHROMA_PERSIST_DIR = "./chroma_db"

# Detect device
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
logger.info(f"Using device: {DEVICE}")

try:
    # Initialize embedding model
    try:
        embedding_model = SentenceTransformer(
            'all-mpnet-base-v2', 
            device=DEVICE
        )
    except Exception as e:
        logger.warning("Using manual model construction due to recursion error")
        word_embedding_model = models.Transformer('sentence-transformers/all-mpnet-base-v2')
        pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension())
        embedding_model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
        embedding_model.to(DEVICE)

    # ChromaDB Client
    chroma_client = chromadb.PersistentClient(path=CHROMA_PERSIST_DIR)
    chroma_embed_fn = embedding_functions.SentenceTransformerEmbeddingFunction(
        model_name="all-mpnet-base-v2"
    )

    # LLM Components
    model_path = "E:\\my python\\chat_bot\\Phi-3-mini-4k-instruct-q4.gguf"
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found: {model_path}")
    
    model = Llama(
        model_path=model_path,
        n_ctx=4096,
        n_threads=8 if torch.cuda.is_available() else 4,
        n_gpu_layers=35 if torch.cuda.is_available() else 0,
        temperature=0.75,
        top_p=0.95,
        repeat_penalty=1.1,
        verbose=False
    )

    logger.info("All components initialized successfully")

except Exception as e:
    logger.error(f"Initialization failed: {e}")
    raise RuntimeError(f"Initialization failed: {e}") from e

# Application State
pdf_collections = {}
chat_histories = {}

def chunk_text(text, chunk_size=MAX_CHUNK_SIZE):
    """Improved chunking with paragraph awareness"""
    chunks = []
    current_chunk = []
    current_length = 0
    
    # Split by paragraphs first
    paragraphs = [p.strip() for p in text.split('\n\n') if p.strip()]
    
    for para in paragraphs:
        para_len = len(para)
        if current_length + para_len > chunk_size:
            chunks.append(' '.join(current_chunk))
            current_chunk = [para]
            current_length = para_len
        else:
            current_chunk.append(para)
            current_length += para_len + 2  # Account for newlines
    
    if current_chunk:
        chunks.append(' '.join(current_chunk))
        
    return chunks

def process_pdf(pdf_file, filename, collection_name):
    """PDF processing with ChromaDB"""
    try:
        # Check existing collection
        try:
            collection = chroma_client.get_collection(collection_name)
            logger.info(f"Using existing collection: {collection_name}")
            return collection
        except Exception:
            logger.info(f"Creating new collection: {collection_name}")
            collection = chroma_client.create_collection(
                name=collection_name,
                embedding_function=chroma_embed_fn
            )

        # Text extraction
        doc = fitz.open(stream=pdf_file, filetype="pdf")
        text = " ".join([page.get_text("text", flags=fitz.TEXT_PRESERVE_LIGATURES) for page in doc])
        text = re.sub(r'[^\x00-\x7F]+', ' ', text)

        # Chunk processing
        chunks = chunk_text(text)
        
        # Batch embeddings
        batch_size = 32
        embeddings = []
        for i in range(0, len(chunks), batch_size):
            batch = chunks[i:i+batch_size]
            embeddings.extend(embedding_model.encode(batch))

        # Batch insertion
        for i in range(0, len(chunks), 100):
            collection.add(
                documents=chunks[i:i+100],
                embeddings=embeddings[i:i+100],
                ids=[f"{collection_name}_chunk_{i+j}" for j in range(len(chunks[i:i+100]))]
            )

        return collection

    except Exception as e:
        logger.error(f"PDF processing error: {e}")
        try:
            chroma_client.delete_collection(collection_name)
        except Exception as cleanup_error:
            logger.error(f"Cleanup failed: {cleanup_error}")
        return None

def answer_question(question, collection):
    """Improved context retrieval"""
    try:
        question_embed = embedding_model.encode([question])
        results = collection.query(
            query_embeddings=question_embed.tolist(),
            n_results=5,  # Get more context
            include=["documents", "distances"]
        )
        
        # Filter by similarity score
        min_distance = 1.5
        relevant_docs = [
            doc for doc, dist in zip(results['documents'][0], results['distances'][0])
            if dist < min_distance
        ]
        
        return "\n\n".join(relevant_docs)[:MAX_CONTEXT_LENGTH]
        
    except Exception as e:
        logger.error(f"Query error: {e}")
        return ""

def generate_answer(question, context, chat_history):
    """Generate detailed answers using enhanced prompting"""
    try:
        # Enhanced prompt template for Phi-3
        history_str = "\n".join(f"### Previous Q: {q}\n### Previous A: {a}" 
                              for q, a in chat_history[-2:])
        
        prompt = f"""<|system|>
        You are a helpful AI assistant. Answer in detail using the context below.
        Provide comprehensive explanations with examples when possible.

        Context:
        {context}

        Chat History:
        {history_str}

        <|user|>
        {question}
        <|assistant|>
        """
        response = model.create_chat_completion(
            messages=[{"role": "user", "content": prompt}],
            max_tokens=2000,
            temperature=0.75,
            top_p=0.95,
            stop=["<|endoftext|>"],
            repeat_penalty=1.1,
            mirostat_mode=2
        )
        
        answer = response['choices'][0]['message']['content'].strip()
        
        # Post-processing for better answers
        min_length = 150
        if len(answer) < min_length:
            return f"{answer}\n\nCould you please elaborate on your question?"
            
        return answer
        
    except Exception as e:
        logger.error(f"Generation error: {e}")
        return "I need more information to answer that properly. Could you rephrase?"

@app.route('/upload', methods=['POST'])
def upload_pdf():
    """Handle PDF uploads with validation"""
    logger.info("Upload endpoint called")
    if 'pdf' not in request.files:
        return jsonify({'error': 'No PDF provided'}), 400

    try:
        file = request.files['pdf']
        pdf_file = file.read()
        
        # File size check
        if len(pdf_file) > MAX_FILE_SIZE:
            return jsonify({'error': 'File too large (max 50MB)'}), 413
            
        filename = file.filename
        
        # Generate collection name
        file_hash = hashlib.md5(pdf_file).hexdigest()
        collection_name = f"{os.path.splitext(filename)[0]}_{file_hash[:8]}"

        # Process with timeout
        with eventlet.Timeout(300, False):
            collection = process_pdf(pdf_file, filename, collection_name)
            if collection:
                pdf_collections[filename] = collection
                chat_histories[filename] = []
                return jsonify({'message': 'PDF processed successfully'}), 200

        # Timeout handling
        logger.error("PDF processing timed out")
        try:
            chroma_client.delete_collection(collection_name)
        except Exception as e:
            logger.error(f"Timeout cleanup failed: {e}")
        return jsonify({'error': 'Processing timed out'}), 408

    except Exception as e:
        logger.error(f"Upload error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/ask', methods=['POST'])
def ask_question():
    """Handle Q&A with improved timeout"""
    logger.info("Ask endpoint called")
    try:
        data = request.get_json()
        question = data.get('question', '').strip()
        filename = data.get('filename')

        if not question:
            return jsonify({'error': 'Empty question'}), 400

        collection = pdf_collections.get(filename)
        if not collection:
            return jsonify({'error': 'Invalid PDF reference'}), 400

        # Process with timeout
        with eventlet.Timeout(60, False):
            context = answer_question(question, collection)
            answer = generate_answer(question, context, chat_histories[filename])
            chat_histories[filename].append((question, answer))
            return jsonify({'answer': answer}), 200

        logger.error("Answer generation timed out")
        return jsonify({'error': 'Answer generation timed out'}), 408

    except Exception as e:
        logger.error(f"Ask error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/')
def serve_index():
    return send_from_directory('templates', 'index.html')

@app.route('/<path:path>')
def serve_static(path):
    return send_from_directory('static', path)

if __name__ == '__main__':
    app.run(host='127.0.0.1', port=5000, debug=True)