# Multimodal Research Paper Assistant (RAG + LLM)

A high-performance, citation-grounded Retrieval-Augmented Generation (RAG) system for scientific literature analysis. This application allows you to upload multiple research papers (PDFs), ask questions with page-level citations, extract figures/tables, search diagrams semantically using CLIP, and perform figure reasoning using vision-language models.

---

## 🚀 Key Features

1. **Grounded Paper Q&A**: Ask complex questions (e.g., "What loss function did they use?", "Summarize Section 4") and receive detailed markdown answers containing direct page citations (e.g., `[Page 7]`).
2. **Multimodal Figure Explorer**: Automatically extracts figures and tables from PDFs, allowing you to browse them in a grid gallery and query them directly (e.g., "Explain the flowchart in Figure 2") using a Vision-Language Model.
3. **Semantic Diagram Retrieval**: Uses CLIP embeddings to index extracted images, enabling you to search for figures using natural language queries (e.g., "transformer encoder block").
4. **Side-by-Side Comparison**: Select any two uploaded papers (e.g., ResNet vs. ViT) and generate a comprehensive comparison table and synthesis report covering architecture, datasets, novelty, and limitations.
5. **Flexible Backend Support**: Toggle seamlessly between Google Gemini API, OpenAI API, and local Ollama deployments directly from the sidebar.

---

## 🛠️ Architecture Workflow

```
                        +---------------------------------------------+
                        |                 User PDF                    |
                        +----------------------+----------------------+
                                               |
                                               v
                                         +-----+-----+
                                         |  PyMuPDF  |
                                         +--+-----+--+
                                            |     |
                          +-----------------+     +-----------------+
                          |                                         |
                          v (Text Extraction)                       v (Figure Extraction)
                    +-----+-----+                             +-----+-----+
                    | Paragraph |                             | Extracted |
                    | Chunking  |                             | PNG Files |
                    +-----+-----+                             +-----+-----+
                          |                                         |
                          v (all-MiniLM-L6-v2)                      v (CLIP ViT-B-32)
                    +-----+-----+                             +-----+-----+
                    | Dense Text|                             | CLIP Image|
                    | Embeddings|                             | Embeddings|
                    +-----+-----+                             +-----+-----+
                          |                                         |
                          v                                         v
                    +-----+-----------------------------------------+-----+
                    |                   FAISS Vector DB                   |
                    +-----+-----------------------------------------+-----+
                          |                                         |
                          | (Query: "Explain Figure 3")             | (Query: "transformer block")
                          v                                         v
                    +-----+-----+                             +-----+-----+
                    | Text RAG  |                             | CLIP Match|
                    | Contexts  |                             |   Image   |
                    +-----+-----+                             +-----+-----+
                          |                                         |
                          +--------------------+--------------------+
                                               |
                                               v
                                     +---------+---------+
                                     |    Gemini / LLM   |
                                     +---------+---------+
                                               |
                                               v
                                     +---------+---------+
                                     |  Grounded Answer  |
                                     |  + Figure Q&A    |
                                     +-------------------+
```

---

## ⚙️ Installation & Setup

### 1. Clone & Setup Directory
Ensure you have Python 3.10+ installed. In your terminal, run:
```bash
# Install dependencies
pip3 install -r requirements.txt
```

### 2. Run the Application
Launch the Streamlit web server:
```bash
streamlit run app.py
```

### 3. Configure API Keys
1. Open the application in your browser (usually `http://localhost:8501`).
2. In the sidebar, select your LLM provider:
   - **Gemini** (Recommended): Input your Google AI Studio API Key. Native multimodal reasoning is supported.
   - **OpenAI**: Input your OpenAI API Key.
   - **Ollama**: Ensure Ollama is running (`ollama serve`). Input your model name (e.g., `llama3` for text or `llava` for vision-language tasks).

---

## 📝 Add This to Your Resume

Include these high-impact bullet points on your AI/ML Resume to showcase your practical engineering capabilities:

* **Built a multimodal Retrieval-Augmented Generation (RAG) system** for scientific literature analysis using PyMuPDF, FAISS vector search, and open-source vision-language models.
* **Implemented dual-semantic indexing (Text + Figures)** using SentenceTransformers and CLIP (`ViT-B-32`), achieving low-latency retrieval of text excerpts and architecture diagrams from research papers.
* **Developed citation-grounded question-answering pipelines** and side-by-side paper comparison matrices, leveraging Google Gemini and OpenAI APIs to produce summaries grounded in specific source page numbers.
* **Designed and deployed a responsive Streamlit dashboard** containing multi-tab workspaces for document library management, interactive RAG Q&A, diagram galleries, and comparative analysis report generation.
