import requests
import json

class LLMClient:
    """Handles standard text generation and basic RAG pipeline prompts."""
    def __init__(self, provider="Ollama", api_key=None, model_name="llama3", ollama_url="http://localhost:11434"):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.ollama_url = ollama_url

    def generate_text(self, prompt, system_instruction=None):
        if self.provider == "Ollama":
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": f"{system_instruction}\n\n{prompt}" if system_instruction else prompt,
                "stream": False
            }
            res = requests.post(url, json=payload)
            return res.json().get("response", "")
        return "API Keys missing. Choose Ollama."

    def generate_rag_answer(self, query, contexts):
        context_str = ""
        for idx, ctx in enumerate(contexts):
            context_str += f"--- CONTEXT BLOCK {idx+1} (Page {ctx['page']}) ---\n{ctx['text']}\n\n"
        sys_inst = "Answer the query using only the provided context blocks. Cite page numbers as [Page X]."
        prompt = f"Query: {query}\n\nContext:\n{context_str}\nAnswer:"
        return self.generate_text(prompt, system_instruction=sys_inst)
