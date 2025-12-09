import fitz  # PyMuPDF
from sentence_transformers import SentenceTransformer
import faiss
import numpy as np

def extract_text_from_pdf(path):
    text = ""
    doc = fitz.open(path)
    for page in doc:
        text += page.get_text()
    return text

path = "/Users/prathamrav/Documents/GitHub/AI-Tutor-Project/syllabus_data/ML/MLMachineLearning-AProbabilisticPerspective.pdf"
text = extract_text_from_pdf(path)
def chunk_text(text, max_words=500):
    words = text.split()
    return [" ".join(words[i:i+max_words]) for i in range(0, len(words), max_words)]

chunks = chunk_text(text)

# Initialize the embedding model
model = SentenceTransformer('all-MiniLM-L6-v2')

# Generate embeddings
embeddings = model.encode(chunks)

# Convert embeddings to numpy
emb_matrix = np.array(embeddings).astype('float32')

# Create FAISS index
index = faiss.IndexFlatL2(emb_matrix.shape[1])
try:
    index.add(emb_matrix)
    print("Embeddings added to FAISS index successfully.")
    # Optionally, check the number of vectors in the index
    print(f"Number of vectors in index: {index.ntotal}")
except Exception as e:
    print(f"Error adding embeddings to FAISS index: {e}")

