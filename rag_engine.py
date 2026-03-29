import os
import re
import fitz # PyMuPDF
import numpy as np
import faiss
import pickle
from sentence_transformers import SentenceTransformer

class PDFProcessor:
    """Handles PDF loading, text extraction, and paragraph segmenting."""
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        
    def extract_text(self, pdf_path):
        doc = fitz.open(pdf_path)
        chunks = []
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            paragraphs = text.split("\n\n")
            for para in paragraphs:
                para = para.strip()
                if para:
                    chunks.append({
                        "text": para,
                        "page": page_num + 1,
                        "doc_name": os.path.basename(pdf_path)
                    })
        doc.close()
        return chunks

class RAGEngine:
    """Manages text embedding, FAISS indexing, and vector search."""
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        self.vector_dir = os.path.join(cache_dir, "vector_store")
        os.makedirs(self.vector_dir, exist_ok=True)
        self.text_index_path = os.path.join(self.vector_dir, "text.index")
        self.text_meta_path = os.path.join(self.vector_dir, "text_metadata.pkl")
        self.text_model = SentenceTransformer("all-MiniLM-L6-v2")
        self.text_chunks = []
        self.text_index = None
        self.load_stores()

    def load_stores(self):
        if os.path.exists(self.text_meta_path):
            with open(self.text_meta_path, "rb") as f:
                self.text_chunks = pickle.load(f)
        if os.path.exists(self.text_index_path):
            self.text_index = faiss.read_index(self.text_index_path)

    def save_stores(self):
        with open(self.text_meta_path, "wb") as f:
            pickle.dump(self.text_chunks, f)
        if self.text_index is not None:
            faiss.write_index(self.text_index, self.text_index_path)

    def process_pdf(self, pdf_path):
        processor = PDFProcessor()
        new_chunks = processor.extract_text(pdf_path)
        new_texts = [c["text"] for c in new_chunks]
        embeddings = self.text_model.encode(new_texts)
        embeddings = np.array(embeddings).astype("float32")
        if self.text_index is None:
            self.text_index = faiss.IndexFlatL2(embeddings.shape[1])
        self.text_index.add(embeddings)
        self.text_chunks.extend(new_chunks)
        self.save_stores()
        return f"Processed {len(new_chunks)} chunks."

    def search_text(self, query, k=5):
        if self.text_index is None:
            return []
        query_emb = self.text_model.encode([query])
        query_emb = np.array(query_emb).astype("float32")
        distances, indices = self.text_index.search(query_emb, k)
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx != -1:
                results.append({
                    "text": self.text_chunks[idx]["text"],
                    "page": self.text_chunks[idx]["page"],
                    "doc_name": self.text_chunks[idx]["doc_name"],
                    "score": float(dist)
                })
        return results
