from sentence_transformers import SentenceTransformer
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import faiss
import numpy as np
import pickle

# Paths to PDF files
file_paths = [
    "/Users/vamshivishruthj/Documents/INTERNSHIP/travelguides/udupi.pdf",
    "/Users/vamshivishruthj/Documents/INTERNSHIP/travelguides/mangalore.pdf",
    "/Users/vamshivishruthj/Documents/INTERNSHIP/travelguides/gokarna.pdf",
    "/Users/vamshivishruthj/Documents/INTERNSHIP/travelguides/coastal.pdf"
]

# Initialize embedding model
embedding_model = SentenceTransformer('all-MiniLM-L6-v2')  # Stable and free

# Step 1: Load and split documents
print("Loading and splitting documents...")
documents = []
for file_path in file_paths:
    loader = PyPDFLoader(file_path)
    documents.extend(loader.load())

text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=100)
chunks = text_splitter.split_documents(documents)

print(f"Created {len(chunks)} chunks.")

# Step 2: Generate embeddings
print("Generating embeddings...")
texts = [chunk.page_content for chunk in chunks]
embeddings = embedding_model.encode(texts)

# Step 3: Create FAISS index and add embeddings
print("Creating FAISS index...")
dimension = embeddings.shape[1]
faiss_index = faiss.IndexFlatL2(dimension)
faiss_index.add(embeddings)

# Step 4: Save FAISS index and text mapping
faiss.write_index(faiss_index, "knowledge_base.index")
with open("text_mapping.pkl", "wb") as f:
    pickle.dump(texts, f)

print("Knowledge base saved successfully.")
