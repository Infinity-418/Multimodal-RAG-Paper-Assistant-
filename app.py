import streamlit as st
import os
from rag_engine import RAGEngine
from llm_client import LLMClient

st.set_page_config(page_title="RAG Research Assistant", layout="wide")
st.title("📚 RAG Research Assistant")

if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

# Sidebar config
provider = st.sidebar.selectbox("LLM Provider", ["Ollama", "Gemini"])
api_key = st.sidebar.text_input("API Key (if Gemini)", type="password")
model_name = st.sidebar.text_input("Model Name", value="llama3")

llm_client = LLMClient(provider=provider, api_key=api_key, model_name=model_name)

# Simple Tabs
tab_upload, tab_chat = st.tabs(["Upload", "Chat Q&A"])

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
