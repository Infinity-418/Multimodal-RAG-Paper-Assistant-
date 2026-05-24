import requests
import json
import base64
from PIL import Image

class LLMClient:
    """Handles standard text generation, basic RAG, and image reasoning."""
    def __init__(self, provider="Ollama", api_key=None, model_name="llama3", ollama_url="http://localhost:11434"):
        self.provider = provider
        self.api_key = api_key
        self.model_name = model_name
        self.ollama_url = ollama_url

    def _encode_image_base64(self, image_path):
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

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
        elif self.provider == "Gemini":
            if not self.api_key:
                return "Gemini key missing."
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            res = model.generate_content(prompt)
            return res.text

    def explain_image(self, image_path, prompt):
        if self.provider == "Gemini":
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            model = genai.GenerativeModel(self.model_name)
            img = Image.open(image_path)
            res = model.generate_content([prompt, img])
            return res.text
        elif self.provider == "Ollama":
            b64_img = self._encode_image_base64(image_path)
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": prompt,
                "images": [b64_img],
                "stream": False
            }
            res = requests.post(url, json=payload)
            return res.json().get("response", "")
        return "Vision support only for Gemini and Ollama."

    def generate_rag_answer(self, query, contexts):
        context_str = ""
        for idx, ctx in enumerate(contexts):
            context_str += f"--- CONTEXT BLOCK {idx+1} (Page {ctx['page']}) ---\n{ctx['text']}\n\n"
        sys_inst = "Answer the query using only the provided context blocks. Cite page numbers as [Page X]."
        prompt = f"Query: {query}\n\nContext:\n{context_str}\nAnswer:"
        return self.generate_text(prompt, system_instruction=sys_inst)
