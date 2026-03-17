import os
import re
import fitz # PyMuPDF
from PIL import Image
import io

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
