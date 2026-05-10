import os
import io
import re
import json
import pickle
import fitz  # PyMuPDF
import numpy as np
import faiss
from PIL import Image
from sentence_transformers import SentenceTransformer

class PDFProcessor:
    """Handles PDF loading, text extraction, text chunking, and image extraction."""
    
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        self.images_dir = os.path.join(cache_dir, "extracted_images")
        os.makedirs(self.images_dir, exist_ok=True)

    def extract_text_and_images(self, pdf_path):
        """
        Extracts text blocks and images from the given PDF.
        Returns:
            chunks (list of dict): Text chunks with text, page number, and document name.
            images_meta (list of dict): Metadata for extracted images.
        """
        doc_name = os.path.basename(pdf_path)
        doc = fitz.open(pdf_path)
        
        chunks = []
        images_meta = []
        
        # 1. Extract and chunk text
        for page_num in range(len(doc)):
            page = doc[page_num]
            text = page.get_text("text")
            
            # Simple text chunking by paragraph or fixed length
            # Let's chunk by paragraphs first (separated by double newlines)
            paragraphs = text.split("\n\n")
            current_chunk = ""
            
            for para in paragraphs:
                para = para.strip()
                if not para:
                    continue
                # If adding this paragraph exceeds 1000 characters, save current chunk and start new one
                if len(current_chunk) + len(para) > 1000 and current_chunk:
                    chunks.append({
                        "text": current_chunk.strip(),
                        "page": page_num + 1,
                        "doc_name": doc_name
                    })
                    # Implement overlapping: keep last 150 chars of previous chunk
                    current_chunk = current_chunk[-150:] + " " + para
                else:
                    current_chunk = (current_chunk + " " + para).strip()
            
            if current_chunk:
                chunks.append({
                    "text": current_chunk.strip(),
                    "page": page_num + 1,
                    "doc_name": doc_name
                })
            
            # 2. Extract Figures/Images
            image_list = page.get_images(full=True)
            for img_idx, img in enumerate(image_list):
                xref = img[0]
                try:
                    base_image = doc.extract_image(xref)
                    image_bytes = base_image["image"]
                    image_ext = base_image["ext"]
                    
                    image = Image.open(io.BytesIO(image_bytes))
                    # Filter out small images (icons, logos, math formulas)
                    if image.width < 150 or image.height < 150:
                        continue
                    
                    # Create clean, unique filename
                    clean_doc_name = re.sub(r'[^a-zA-Z0-9_.-]', '_', doc_name)
                    img_filename = f"{clean_doc_name}_p{page_num+1}_img{img_idx+1}.{image_ext}"
                    img_path = os.path.join(self.images_dir, img_filename)
                    
                    image.save(img_path)
                    
                    # Extract surrounding text as a caption candidate
                    # We will grab lines on the same page containing "Figure" or "Fig"
                    caption = self._find_caption_on_page(text, img_idx + 1, page_num + 1)
                    
                    images_meta.append({
                        "path": os.path.abspath(img_path),
                        "filename": img_filename,
                        "page": page_num + 1,
                        "doc_name": doc_name,
                        "img_idx": img_idx + 1,
                        "caption": caption
                    })
                except Exception as e:
                    print(f"Error extracting image {xref} on page {page_num+1}: {e}")
                    
        doc.close()
        return chunks, images_meta

    def _find_caption_on_page(self, page_text, img_num, page_num):
        """Helper to search page text for figure captions."""
        # Find lines starting with Figure/Fig. followed by number
        pattern = rf"(?:Figure|Fig\.)\s*(?:{img_num}|\d+)[^\n]+"
        matches = re.findall(pattern, page_text, re.IGNORECASE)
        if matches:
            return matches[0].strip()
        # Fallback search for any figure label
        pattern_any = r"(?:Figure|Fig\.)\s*\d+[^\n]+"
        matches_any = re.findall(pattern_any, page_text, re.IGNORECASE)
        if matches_any:
            # Return first one as a best guess
            return matches_any[0].strip()
        return f"Figure from page {page_num}"


class RAGEngine:
    """Manages text and image embedding, FAISS indexing, caching, and retrieval."""
    
    def __init__(self, cache_dir=".cache"):
        self.cache_dir = cache_dir
        self.vector_dir = os.path.join(cache_dir, "vector_store")
        os.makedirs(self.vector_dir, exist_ok=True)
        
        # Paths for serialized indexes and metadata
        self.text_index_path = os.path.join(self.vector_dir, "text.index")
        self.text_meta_path = os.path.join(self.vector_dir, "text_metadata.pkl")
        self.image_index_path = os.path.join(self.vector_dir, "image.index")
        self.image_meta_path = os.path.join(self.vector_dir, "image_metadata.pkl")
        
        # Initialize Embedding Models (Lazy loading)
        self._text_model = None
        self._clip_model = None
        
        # Load or initialize metadata and indexes
        self.text_chunks = []
        self.images_meta = []
        
        self.text_index = None
        self.image_index = None
        
        self.load_stores()

    @property
    def text_model(self):
        if self._text_model is None:
            # Fast, high-quality, small embedding model
            self._text_model = SentenceTransformer("all-MiniLM-L6-v2")
        return self._text_model

    @property
    def clip_model(self):
        if self._clip_model is None:
            # CLIP model for multimodal embedding (text and image)
            self._clip_model = SentenceTransformer("clip-ViT-B-32")
        return self._clip_model

    def load_stores(self):
        """Loads FAISS indices and metadata from disk if they exist."""
        # Load text chunks metadata
        if os.path.exists(self.text_meta_path):
            with open(self.text_meta_path, "rb") as f:
                self.text_chunks = pickle.load(f)
                
        # Load text FAISS index
        if os.path.exists(self.text_index_path):
            self.text_index = faiss.read_index(self.text_index_path)
            
        # Load image metadata
        if os.path.exists(self.image_meta_path):
            with open(self.image_meta_path, "rb") as f:
                self.images_meta = pickle.load(f)
                
        # Load image FAISS index
        if os.path.exists(self.image_index_path):
            self.image_index = faiss.read_index(self.image_index_path)

    def save_stores(self):
        """Saves FAISS indices and metadata to disk."""
        # Save text metadata
        with open(self.text_meta_path, "wb") as f:
            pickle.dump(self.text_chunks, f)
        # Save text index
        if self.text_index is not None:
            faiss.write_index(self.text_index, self.text_index_path)
            
        # Save image metadata
        with open(self.image_meta_path, "wb") as f:
            pickle.dump(self.images_meta, f)
        # Save image index
        if self.image_index is not None:
            faiss.write_index(self.image_index, self.image_index_path)

    def process_pdf(self, pdf_path):
        """Parses a new PDF, extracts text/images, computes embeddings, and updates indices."""
        doc_name = os.path.basename(pdf_path)
        
        # Check if already processed
        already_processed = any(chunk["doc_name"] == doc_name for chunk in self.text_chunks)
        if already_processed:
            return f"'{doc_name}' is already processed."
            
        processor = PDFProcessor(cache_dir=self.cache_dir)
        new_chunks, new_images = processor.extract_text_and_images(pdf_path)
        
        if not new_chunks:
            return f"No text could be extracted from '{doc_name}'."
            
        # 1. Update text FAISS index
        new_texts = [chunk["text"] for chunk in new_chunks]
        new_text_embeddings = self.text_model.encode(new_texts, show_progress_bar=False)
        new_text_embeddings = np.array(new_text_embeddings).astype("float32")
        
        if self.text_index is None:
            dim = new_text_embeddings.shape[1]
            self.text_index = faiss.IndexFlatL2(dim)
            
        self.text_index.add(new_text_embeddings)
        self.text_chunks.extend(new_chunks)
        
        # 2. Update image FAISS index (if any images were extracted)
        if new_images:
            new_pil_images = []
            valid_images_meta = []
            
            for img_meta in new_images:
                try:
                    img = Image.open(img_meta["path"])
                    new_pil_images.append(img)
                    valid_images_meta.append(img_meta)
                except Exception as e:
                    print(f"Error loading image {img_meta['path']} for embedding: {e}")
                    
            if new_pil_images:
                # Encode images using CLIP
                new_img_embeddings = self.clip_model.encode(new_pil_images, show_progress_bar=False)
                new_img_embeddings = np.array(new_img_embeddings).astype("float32")
                
                if self.image_index is None:
                    dim_img = new_img_embeddings.shape[1]
                    self.image_index = faiss.IndexFlatL2(dim_img)
                    
                self.image_index.add(new_img_embeddings)
                self.images_meta.extend(valid_images_meta)
                
        # Save updated indices to disk
        self.save_stores()
        
        return f"Successfully processed '{doc_name}'. Extracted {len(new_chunks)} text chunks and {len(new_images)} figures."

    def search_text(self, query, k=5, filter_doc=None):
        """
        Searches for top k text chunks matching the query.
        Optionally filters by document name.
        """
        if self.text_index is None or not self.text_chunks:
            return []
            
        # Encode query
        query_embedding = self.text_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        
        # Search index
        # To support filtering, we search for more than k and filter manually if filter_doc is provided
        search_k = k * 5 if filter_doc else k
        distances, indices = self.text_index.search(query_embedding, min(search_k, len(self.text_chunks)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.text_chunks):
                continue
            chunk = self.text_chunks[idx]
            if filter_doc and chunk["doc_name"] != filter_doc:
                continue
            
            results.append({
                "text": chunk["text"],
                "page": chunk["page"],
                "doc_name": chunk["doc_name"],
                "score": float(dist)
            })
            if len(results) >= k:
                break
                
        return results

    def search_images(self, query, k=3, filter_doc=None):
        """
        Searches for top k images matching the text query using CLIP.
        """
        if self.image_index is None or not self.images_meta:
            return []
            
        # Encode query text using CLIP
        query_embedding = self.clip_model.encode([query])
        query_embedding = np.array(query_embedding).astype("float32")
        
        # Search index
        search_k = k * 5 if filter_doc else k
        distances, indices = self.image_index.search(query_embedding, min(search_k, len(self.images_meta)))
        
        results = []
        for dist, idx in zip(distances[0], indices[0]):
            if idx == -1 or idx >= len(self.images_meta):
                continue
            img_meta = self.images_meta[idx]
            if filter_doc and img_meta["doc_name"] != filter_doc:
                continue
                
            results.append({
                "path": img_meta["path"],
                "filename": img_meta["filename"],
                "page": img_meta["page"],
                "doc_name": img_meta["doc_name"],
                "img_idx": img_meta["img_idx"],
                "caption": img_meta["caption"],
                "score": float(dist)
            })
            if len(results) >= k:
                break
                
        return results

    def get_all_documents(self):
        """Returns list of unique document names currently in index."""
        return list(set(chunk["doc_name"] for chunk in self.text_chunks))

    def get_document_images(self, doc_name):
        """Returns all images extracted from a specific document."""
        return [img for img in self.images_meta if img["doc_name"] == doc_name]

    def remove_document(self, doc_name):
        """Removes a document from the index and metadata, then rebuilds indices."""
        # Find which chunks to keep
        keep_text_indices = [i for i, chunk in enumerate(self.text_chunks) if chunk["doc_name"] != doc_name]
        keep_image_indices = [i for i, img in enumerate(self.images_meta) if img["doc_name"] != doc_name]
        
        # Update text structures
        if len(keep_text_indices) == len(self.text_chunks):
            # No changes needed
            return
            
        # Rebuild text index
        new_text_chunks = [self.text_chunks[i] for i in keep_text_indices]
        if new_text_chunks:
            new_texts = [c["text"] for c in new_text_chunks]
            new_embeddings = self.text_model.encode(new_texts, show_progress_bar=False)
            new_embeddings = np.array(new_embeddings).astype("float32")
            dim = new_embeddings.shape[1]
            self.text_index = faiss.IndexFlatL2(dim)
            self.text_index.add(new_embeddings)
            self.text_chunks = new_text_chunks
        else:
            self.text_chunks = []
            self.text_index = None
            
        # Rebuild image index
        new_images_meta = [self.images_meta[i] for i in keep_image_indices]
        if new_images_meta:
            new_pil_images = []
            for img in new_images_meta:
                new_pil_images.append(Image.open(img["path"]))
            new_img_embeddings = self.clip_model.encode(new_pil_images, show_progress_bar=False)
            new_img_embeddings = np.array(new_img_embeddings).astype("float32")
            dim_img = new_img_embeddings.shape[1]
            self.image_index = faiss.IndexFlatL2(dim_img)
            self.image_index.add(new_img_embeddings)
            self.images_meta = new_images_meta
            
            # Delete physical image files of the removed document
            for img in self.images_meta:
                if img["doc_name"] == doc_name:
                    try:
                        os.remove(img["path"])
                    except:
                        pass
        else:
            self.images_meta = []
            self.image_index = None
            
        self.save_stores()
