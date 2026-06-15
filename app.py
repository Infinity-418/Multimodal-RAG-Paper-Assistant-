import os
import shutil
import streamlit as st
from PIL import Image

# Import custom modules
from rag_engine import RAGEngine
from llm_client import LLMClient

# Setup Page Config
st.set_page_config(
    page_title="Multimodal Research RAG Assistant",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom Premium Styling
CUSTOM_CSS = """
<style>
    /* Import Google Fonts */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=Outfit:wght@400;500;600;700;800&display=swap');
    
    /* Global Overrides */
    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }
    
    h1, h2, h3, h4 {
        font-family: 'Outfit', sans-serif;
        font-weight: 700;
    }
    
    /* Header Gradient */
    .main-header {
        background: linear-gradient(135deg, #6366f1 0%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        font-family: 'Outfit', sans-serif;
        font-weight: 800;
        font-size: 2.8rem;
        margin-bottom: 0.2rem;
    }
    
    .subtitle {
        font-size: 1.1rem;
        color: #9ca3af;
        margin-bottom: 2rem;
        font-weight: 400;
    }
    
    /* Card Styles */
    .glass-card {
        background: rgba(17, 24, 39, 0.45);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 12px;
        padding: 1.5rem;
        margin-bottom: 1rem;
        box-shadow: 0 4px 20px 0 rgba(0, 0, 0, 0.15);
        transition: transform 0.2s ease, border-color 0.2s ease;
    }
    .glass-card:hover {
        border-color: rgba(99, 102, 241, 0.4);
        transform: translateY(-2px);
    }
    
    /* Badge Styles */
    .citation-badge {
        display: inline-block;
        background: rgba(99, 102, 241, 0.15);
        color: #818cf8;
        border: 1px solid rgba(99, 102, 241, 0.3);
        padding: 0.2rem 0.6rem;
        border-radius: 6px;
        font-size: 0.8rem;
        font-weight: 500;
        margin-right: 0.5rem;
        margin-bottom: 0.5rem;
    }
    
    /* Chat Bubble Design */
    .chat-bubble {
        padding: 1rem 1.25rem;
        border-radius: 12px;
        margin-bottom: 1rem;
        line-height: 1.5;
        font-size: 0.95rem;
    }
    .chat-user {
        background-color: #1e1b4b;
        border: 1px solid #312e81;
        color: #e0e7ff;
        margin-left: 2rem;
    }
    .chat-assistant {
        background-color: #0f172a;
        border: 1px solid #1e293b;
        color: #f1f5f9;
        margin-right: 2rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
    }
    
    /* Gallery Hover Zoom */
    .gallery-img-container {
        position: relative;
        overflow: hidden;
        border-radius: 8px;
        border: 1px solid rgba(255,255,255,0.08);
        background: #111827;
        margin-bottom: 10px;
    }
    .gallery-img-container img {
        transition: transform 0.3s ease;
    }
    .gallery-img-container:hover img {
        transform: scale(1.05);
    }
</style>
"""
st.markdown(CUSTOM_CSS, unsafe_allow_html=True)

# ----------------- SESSION STATE SETUP -----------------
if "rag_engine" not in st.session_state:
    with st.spinner("Initializing AI Embeddings & Vector Storage..."):
        st.session_state.rag_engine = RAGEngine()

if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

if "summaries" not in st.session_state:
    st.session_state.summaries = {}

# Directory to store uploaded PDFs physically
UPLOAD_DIR = "uploaded_papers"
os.makedirs(UPLOAD_DIR, exist_ok=True)


# ----------------- SIDEBAR: CONFIGURATION -----------------
st.sidebar.markdown("<h2 style='font-family:Outfit; margin-bottom: 0.5rem;'>⚙️ Settings</h2>", unsafe_allow_html=True)

# Model Provider Selection
provider = st.sidebar.selectbox(
    "LLM Provider",
    ["Gemini", "OpenAI", "Ollama"],
    index=0,
    help="Select the AI service provider. Gemini is highly recommended for multimodal use cases."
)

# API Keys and Model Names
api_key = ""
model_name = ""
ollama_url = "http://localhost:11434"

if provider == "Gemini":
    # Check environment variable as fallback
    default_key = os.environ.get("GEMINI_API_KEY", "")
    api_key = st.sidebar.text_input("Gemini API Key", type="password", value=default_key, help="Enter your Google AI Studio API key.")
    model_name = st.sidebar.selectbox("Model", ["gemini-1.5-flash", "gemini-2.0-flash", "gemini-1.5-pro", "gemini-2.5-flash"], index=0)
    
elif provider == "OpenAI":
    default_key = os.environ.get("OPENAI_API_KEY", "")
    api_key = st.sidebar.text_input("OpenAI API Key", type="password", value=default_key, help="Enter your OpenAI API key.")
    model_name = st.sidebar.selectbox("Model", ["gpt-4o-mini", "gpt-4o"], index=0)
    
elif provider == "Ollama":
    ollama_url = st.sidebar.text_input("Ollama Host URL", value="http://localhost:11434")
    model_name = st.sidebar.text_input("Model Name (e.g. llama3, qwen2.5, llava)", value="llama3")

# Instantiate LLM Client
llm_client = None
if provider == "Ollama" or (provider in ["Gemini", "OpenAI"] and api_key):
    llm_client = LLMClient(provider=provider, api_key=api_key, model_name=model_name, ollama_url=ollama_url)
else:
    st.sidebar.warning("⚠️ Please provide an API key to enable the AI assistant features.")

st.sidebar.markdown("---")
st.sidebar.markdown("<h3 style='font-family:Outfit;'>📊 Index Statistics</h3>", unsafe_allow_html=True)

# Compute statistics
documents = st.session_state.rag_engine.get_all_documents()
num_docs = len(documents)
num_chunks = len(st.session_state.rag_engine.text_chunks)
num_images = len(st.session_state.rag_engine.images_meta)

stats_html = f"""
<div style="background: rgba(255,255,255,0.03); padding: 10px; border-radius: 8px; font-size: 0.9rem;">
    <div>📄 <b>Documents Indexed:</b> {num_docs}</div>
    <div>🧩 <b>Text Chunks:</b> {num_chunks}</div>
    <div>🖼️ <b>Figures Extracted:</b> {num_images}</div>
</div>
"""
st.sidebar.markdown(stats_html, unsafe_allow_html=True)

# Clear Index Button
st.sidebar.markdown("<br>", unsafe_allow_html=True)
if st.sidebar.button("🗑️ Clear Vector Database", use_container_width=True):
    # Wipe the directories
    if os.path.exists(".cache"):
        shutil.rmtree(".cache")
    if os.path.exists(UPLOAD_DIR):
        shutil.rmtree(UPLOAD_DIR)
    os.makedirs(UPLOAD_DIR, exist_ok=True)
    
    st.session_state.rag_engine = RAGEngine()
    st.session_state.chat_history = []
    st.sidebar.success("Database cleared successfully!")
    st.rerun()


# ----------------- MAIN PANEL -----------------
st.markdown("<div class='main-header'>Multimodal Research Paper Assistant</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>Analyze scientific literature, extract architecture diagrams, and perform citation-grounded RAG.</div>", unsafe_allow_html=True)

# Create layout tabs
tab_ingest, tab_chat, tab_figures, tab_compare = st.tabs([
    "📚 Library & Import", 
    "💬 Paper Chat (Q&A)", 
    "🎨 Figures & Visuals", 
    "⚖️ Paper Comparison"
])


# ----------------- TAB 1: LIBRARY & IMPORT -----------------
with tab_ingest:
    st.markdown("<h3 style='font-family:Outfit;'>📥 Import Research Papers</h3>", unsafe_allow_html=True)
    st.write("Upload PDF versions of research papers from arXiv, PubMed, or local files to index them in the FAISS database.")
    
    uploaded_files = st.file_uploader(
        "Choose PDF files", 
        type=["pdf"], 
        accept_multiple_files=True,
        label_visibility="collapsed"
    )
    
    if uploaded_files:
        if st.button("🚀 Process & Index Uploaded Papers", type="primary"):
            for uploaded_file in uploaded_files:
                # Save locally
                file_path = os.path.join(UPLOAD_DIR, uploaded_file.name)
                with open(file_path, "wb") as f:
                    f.write(uploaded_file.getbuffer())
                
                # Ingest to FAISS
                with st.spinner(f"Ingesting '{uploaded_file.name}' (extracting text, images, and generating vector embeddings)..."):
                    result = st.session_state.rag_engine.process_pdf(file_path)
                    st.success(result)
            st.rerun()

    st.markdown("<br><hr><br>", unsafe_allow_html=True)
    st.markdown("<h3 style='font-family:Outfit;'>📚 Library Explorer</h3>", unsafe_allow_html=True)
    
    if num_docs == 0:
        st.info("Your library is currently empty. Upload PDFs above to begin.")
    else:
        # Show table of documents
        for doc in documents:
            doc_chunks = [c for c in st.session_state.rag_engine.text_chunks if c["doc_name"] == doc]
            doc_imgs = [img for img in st.session_state.rag_engine.images_meta if img["doc_name"] == doc]
            pages_count = max(c["page"] for c in doc_chunks) if doc_chunks else 0
            
            col_info, col_sum, col_del = st.columns([5, 1.2, 1])
            
            with col_info:
                st.markdown(f"""
                <div class="glass-card">
                    <div style="font-weight: 600; font-size: 1.1rem; color: #818cf8;">📄 {doc}</div>
                    <div style="font-size: 0.85rem; color: #9ca3af; margin-top: 5px;">
                        Pages: {pages_count} &nbsp;|&nbsp; 
                        Chunks: {len(doc_chunks)} &nbsp;|&nbsp; 
                        Extracted Figures: {len(doc_imgs)}
                    </div>
                </div>
                """, unsafe_allow_html=True)
                
            with col_sum:
                st.write("")  # Padding
                st.write("")
                if st.button("📝 Summarize", key=f"sum_{doc}", use_container_width=True):
                    if not llm_client:
                        st.error("Please configure API credentials in the sidebar.")
                    else:
                        with st.spinner(f"Generating summary for '{doc}'..."):
                            try:
                                summary_text = llm_client.generate_summary(doc, doc_chunks)
                                st.session_state.summaries[doc] = summary_text
                                st.rerun()
                            except Exception as e:
                                st.error(f"Error generating summary: {e}")

            with col_del:
                st.write("")  # Padding
                st.write("")
                if st.button("🗑️ Remove", key=f"del_{doc}", use_container_width=True):
                    st.session_state.rag_engine.remove_document(doc)
                    # Delete physical file
                    phys_path = os.path.join(UPLOAD_DIR, doc)
                    if os.path.exists(phys_path):
                        os.remove(phys_path)
                    # Clear summary if exists
                    if doc in st.session_state.summaries:
                        del st.session_state.summaries[doc]
                    st.success(f"Removed '{doc}' from vector index.")
                    st.rerun()
            
            # Render summary card if exists
            if doc in st.session_state.summaries:
                with st.expander(f"📝 Executive Summary for '{doc}'", expanded=True):
                    st.markdown(st.session_state.summaries[doc])


# ----------------- TAB 2: PAPER CHAT (Q&A) -----------------
with tab_chat:
    st.markdown("<h3 style='font-family:Outfit;'>💬 Grounded Question Answering</h3>", unsafe_allow_html=True)
    
    # Active Doc Selection
    doc_options = ["All Indexed Papers"] + documents
    selected_doc = st.selectbox(
        "Focus Search Context", 
        doc_options, 
        index=0,
        help="Select a specific paper to restrict search context, or search across your entire library."
    )
    filter_doc_name = None if selected_doc == "All Indexed Papers" else selected_doc

    # Quick Suggestion Chips
    st.write("💡 *Try asking:*")
    col_s1, col_s2, col_s3 = st.columns(3)
    
    # Check prompt submissions from quick suggestions
    suggested_query = None
    if col_s1.button("What is the core methodology and architecture proposed?", use_container_width=True):
        suggested_query = "What is the core methodology and architecture proposed?"
    if col_s2.button("What dataset and evaluation metrics were used?", use_container_width=True):
        suggested_query = "What dataset and evaluation metrics were used?"
    if col_s3.button("What limitations or future scope are mentioned?", use_container_width=True):
        suggested_query = "What limitations or future scope are mentioned?"

    st.markdown("<br>", unsafe_allow_html=True)
    
    # Conversation Panel
    for chat in st.session_state.chat_history:
        role = chat["role"]
        content = chat["content"]
        
        if role == "user":
            st.markdown(f'<div class="chat-bubble chat-user">🧑‍💻 <b>You:</b><br>{content}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="chat-bubble chat-assistant">🤖 <b>Assistant:</b><br>{content}</div>', unsafe_allow_html=True)
            
            # Show citations and diagrams if present
            if "citations" in chat and chat["citations"]:
                with st.expander("📌 Citation Grounding (Retrieved Text Sources)"):
                    for idx, cit in enumerate(chat["citations"]):
                        st.markdown(f"""
                        <div style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-left: 3px solid #6366f1; margin-bottom: 8px; border-radius: 0 6px 6px 0; font-size: 0.85rem;">
                            <span class="citation-badge">Page {cit['page']}</span> 
                            <b>{cit['doc_name']}</b> (Score: {cit['score']:.4f})
                            <div style="color: #cbd5e1; margin-top: 5px; font-style: italic;">"{cit['text']}"</div>
                        </div>
                        """, unsafe_allow_html=True)
            
            if "images" in chat and chat["images"]:
                with st.expander("🖼️ Relevant Figures Matched (via CLIP Semantic Search)"):
                    for img in chat["images"]:
                        col_img_view, col_img_desc = st.columns([1, 2])
                        with col_img_view:
                            if os.path.exists(img["path"]):
                                st.image(img["path"], use_container_width=True)
                        with col_img_desc:
                            st.markdown(f"""
                            <b>{img['filename']}</b><br>
                            📄 Page: {img['page']} &nbsp;|&nbsp; 🔍 Clip Score: {img['score']:.4f}<br>
                            📝 <i>Caption: {img['caption']}</i>
                            """, unsafe_allow_html=True)

    # Chat Input
    query_input = st.chat_input("Ask a question about the papers...")
    
    # If a quick suggestion chip was clicked, override the query input
    query = query_input or suggested_query
    
    if query:
        if not llm_client:
            st.error("Please configure a Model Provider and API Key in the sidebar to run the chat.")
        elif num_docs == 0:
            st.warning("Please upload and process at least one paper in the 'Library' tab before asking questions.")
        else:
            # Append User message to history
            st.session_state.chat_history.append({"role": "user", "content": query})
            
            # Show User message immediately
            st.markdown(f'<div class="chat-bubble chat-user">🧑‍💻 <b>You:</b><br>{query}</div>', unsafe_allow_html=True)
            
            # Retrieve Relevant Text Chunks
            with st.spinner("Searching document index..."):
                retrieved_chunks = st.session_state.rag_engine.search_text(query, k=5, filter_doc=filter_doc_name)
                
                # Retrieve Relevant Figures (CLIP Multimodal Search)
                matched_images = st.session_state.rag_engine.search_images(query, k=2, filter_doc=filter_doc_name)
                # Filter images based on CLIP Cosine Similarity threshold (higher is better, >= 0.22 indicates high relevance)
                matched_images = [img for img in matched_images if img["score"] > 0.22]
            
            # Generate Answer
            with st.spinner("Synthesizing answer with citations..."):
                try:
                    answer = llm_client.generate_rag_answer(query, retrieved_chunks)
                    
                    # Store response in session state
                    chat_response = {
                        "role": "assistant",
                        "content": answer,
                        "citations": retrieved_chunks,
                        "images": matched_images
                    }
                    st.session_state.chat_history.append(chat_response)
                    
                    # Display response
                    st.markdown(f'<div class="chat-bubble chat-assistant">🤖 <b>Assistant:</b><br>{answer}</div>', unsafe_allow_html=True)
                    
                    # Show citations and diagrams
                    if retrieved_chunks:
                        with st.expander("📌 Citation Grounding (Retrieved Text Sources)", expanded=True):
                            for idx, cit in enumerate(retrieved_chunks):
                                st.markdown(f"""
                                <div style="background: rgba(255,255,255,0.02); padding: 8px 12px; border-left: 3px solid #6366f1; margin-bottom: 8px; border-radius: 0 6px 6px 0; font-size: 0.85rem;">
                                    <span class="citation-badge">Page {cit['page']}</span> 
                                    <b>{cit['doc_name']}</b> (Score: {cit['score']:.4f})
                                    <div style="color: #cbd5e1; margin-top: 5px; font-style: italic;">"{cit['text']}"</div>
                                </div>
                                """, unsafe_allow_html=True)
                                
                    if matched_images:
                        with st.expander("🖼️ Relevant Figures Matched (via CLIP Semantic Search)", expanded=True):
                            for img in matched_images:
                                col_img_view, col_img_desc = st.columns([1, 2])
                                with col_img_view:
                                    if os.path.exists(img["path"]):
                                        st.image(img["path"], use_container_width=True)
                                with col_img_desc:
                                    st.markdown(f"""
                                    <b>{img['filename']}</b><br>
                                    📄 Page: {img['page']} &nbsp;|&nbsp; 🔍 Clip Score: {img['score']:.4f}<br>
                                    📝 <i>Caption: {img['caption']}</i>
                                    """, unsafe_allow_html=True)
                    
                    # Removed redundant st.rerun() to avoid chat bubble duplications
                except Exception as e:
                    st.error(f"Error generating response: {e}")


# ----------------- TAB 3: FIGURES & VISUALS -----------------
with tab_figures:
    st.markdown("<h3 style='font-family:Outfit;'>🎨 Multimodal Figure Explorer</h3>", unsafe_allow_html=True)
    
    if num_docs == 0:
        st.info("Your library is empty. Please upload and process a PDF document containing images.")
    else:
        selected_fig_doc = st.selectbox(
            "Select Paper to View Figures", 
            documents,
            key="fig_doc_select"
        )
        
        doc_images = st.session_state.rag_engine.get_document_images(selected_fig_doc)
        
        if not doc_images:
            st.warning(f"No figures/images of significant size (>=150x150) were extracted from '{selected_fig_doc}'.")
        else:
            st.write(f"Found **{len(doc_images)}** figures in this document. Select a figure below to ask questions about its layout, charts, or diagram architecture.")
            
            # Render a grid of thumbnails
            cols_per_row = 4
            for i in range(0, len(doc_images), cols_per_row):
                cols = st.columns(cols_per_row)
                for j in range(cols_per_row):
                    img_idx = i + j
                    if img_idx < len(doc_images):
                        img_data = doc_images[img_idx]
                        with cols[j]:
                            # Gallery thumbnail card
                            st.markdown(f"<div class='gallery-img-container'>", unsafe_allow_html=True)
                            if os.path.exists(img_data["path"]):
                                st.image(img_data["path"], use_container_width=True)
                            st.markdown("</div>", unsafe_allow_html=True)
                            st.markdown(f"<b>Figure {img_data['img_idx']}</b> (Page {img_data['page']})", unsafe_allow_html=True)
                            st.caption(f"{img_data['caption'][:60]}...")
            
            st.markdown("<br><hr><br>", unsafe_allow_html=True)
            st.markdown("<h4 style='font-family:Outfit;'>💬 Query Selected Figure (Multimodal Q&A)</h4>", unsafe_allow_html=True)
            
            # Let user select figure by index
            fig_options = [f"Figure {img['img_idx']} (Page {img['page']})" for img in doc_images]
            selected_fig_idx = st.selectbox("Choose Figure to Query", range(len(doc_images)), format_func=lambda x: fig_options[x])
            
            active_fig = doc_images[selected_fig_idx]
            
            col_active_img, col_active_qa = st.columns([1, 1])
            
            with col_active_img:
                st.markdown("<div class='glass-card' style='text-align: center;'>", unsafe_allow_html=True)
                if os.path.exists(active_fig["path"]):
                    st.image(active_fig["path"], caption=active_fig["caption"], use_container_width=True)
                st.markdown("</div>", unsafe_allow_html=True)
                
            with col_active_qa:
                st.write("🤖 **Vision LLM Figure Reasoning**")
                st.info(f"Context: Page {active_fig['page']} of {active_fig['doc_name']}")
                
                # Check for multimodal provider validation
                if provider == "Ollama" and "llava" not in model_name.lower():
                    st.warning("⚠️ For local figure Q&A, please load a vision-capable model in Ollama (e.g. `llava` or `bakllava`) and input its name in the sidebar.")
                
                fig_prompt = st.text_area(
                    "What would you like to know about this figure?",
                    value="Explain this flowchart/diagram in detail, explaining what each component does.",
                    height=100
                )
                
                if st.button("🔍 Analyze Figure", type="primary"):
                    if not llm_client:
                        st.error("Please configure API credentials in the sidebar.")
                    else:
                        with st.spinner("Analyzing image..."):
                            try:
                                analysis_result = llm_client.explain_image(active_fig["path"], fig_prompt)
                                st.markdown("<div class='glass-card' style='background: rgba(99,102,241,0.05); border-color: rgba(99,102,241,0.2);'>", unsafe_allow_html=True)
                                st.markdown(f"**Analysis:**\n\n{analysis_result}")
                                st.markdown("</div>", unsafe_allow_html=True)
                            except Exception as e:
                                st.error(f"Error calling vision model: {e}")


# ----------------- TAB 4: PAPER COMPARISON -----------------
with tab_compare:
    st.markdown("<h3 style='font-family:Outfit;'>⚖️ Side-by-Side Paper Comparison</h3>", unsafe_allow_html=True)
    st.write("Select two papers from your library to compare their methodologies, experiments, architecture design, and results.")
    
    if len(documents) < 2:
        st.info("Please index at least 2 research papers to enable the comparison feature.")
    else:
        col_p1, col_p2 = st.columns(2)
        
        with col_p1:
            paper_a = st.selectbox("Select Paper A", documents, index=0)
        with col_p2:
            paper_b = st.selectbox("Select Paper B", documents, index=1 if len(documents) > 1 else 0)
            
        if paper_a == paper_b:
            st.warning("Please select two different papers to compare.")
        else:
            if st.button("⚖️ Generate Comparative Analysis", type="primary", use_container_width=True):
                if not llm_client:
                    st.error("Please configure your API Key in the sidebar.")
                else:
                    with st.spinner(f"Reading and analyzing '{paper_a}' and '{paper_b}'..."):
                        # Get chunks for both documents
                        doc1_chunks = [c for c in st.session_state.rag_engine.text_chunks if c["doc_name"] == paper_a]
                        doc2_chunks = [c for c in st.session_state.rag_engine.text_chunks if c["doc_name"] == paper_b]
                        
                        try:
                            comparison_report = llm_client.generate_comparison(
                                paper_a, doc1_chunks, 
                                paper_b, doc2_chunks
                            )
                            
                            st.markdown("<br><hr><br>", unsafe_allow_html=True)
                            st.markdown("<h4 style='font-family:Outfit;'>📄 Comparative Analysis Report</h4>", unsafe_allow_html=True)
                            
                            st.markdown(comparison_report)
                            
                            # Add download button
                            st.download_button(
                                label="📥 Download Comparison Report (.md)",
                                data=comparison_report,
                                file_name=f"comparison_{paper_a.split('.')[0]}_vs_{paper_b.split('.')[0]}.md",
                                mime="text/markdown"
                            )
                        except Exception as e:
                            st.error(f"Error generating comparison: {e}")
