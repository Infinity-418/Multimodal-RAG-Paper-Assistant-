import streamlit as st
import os
from rag_engine import RAGEngine
from llm_client import LLMClient

st.set_page_config(page_title="Multimodal RAG Assistant", layout="wide")
st.title("📚 Multimodal Research RAG")

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

# Sidebar config
provider = st.sidebar.selectbox("LLM Provider", ["Ollama", "Gemini"])
api_key = st.sidebar.text_input("API Key (if Gemini)", type="password")
model_name = st.sidebar.text_input("Model Name", value="llama3")

llm_client = LLMClient(provider=provider, api_key=api_key, model_name=model_name)

# Three Tabs (added figures explorer)
tab_upload, tab_chat, tab_figures = st.tabs(["Upload & Library", "Chat Q&A", "Figures Explorer"])

with tab_upload:
    uploaded = st.file_uploader("Upload Paper", type=["pdf"])
    if uploaded:
        path = os.path.join(".cache", uploaded.name)
        with open(path, "wb") as f:
            f.write(uploaded.getbuffer())
        res = st.session_state.rag_engine.process_pdf(path)
        st.success(res)

with tab_chat:
    query = st.text_input("Ask a question about the papers:")
    if query:
        contexts = st.session_state.rag_engine.search_text(query)
        ans = llm_client.generate_rag_answer(query, contexts)
        st.write(ans)

with tab_figures:
    documents = st.session_state.rag_engine.get_all_documents()
    if documents:
        doc = st.selectbox("Select document", documents)
        imgs = st.session_state.rag_engine.get_document_images(doc)
        if imgs:
            for img in imgs:
                st.image(img["path"], caption=img["caption"])
