import os
import base64
import requests
import json
from PIL import Image

class LLMClient:
    """Universal client for OpenAI, Google Gemini, and local Ollama APIs."""
    
    def __init__(self, provider="Gemini", api_key=None, model_name=None, ollama_url="http://localhost:11434"):
        self.provider = provider
        self.api_key = api_key
        self.ollama_url = ollama_url
        
        # Set default models based on provider
        if provider == "Gemini":
            self.model_name = model_name or "gemini-1.5-flash"
            # Lazy initialize later inside query methods to avoid importing when not used
        elif provider == "OpenAI":
            self.model_name = model_name or "gpt-4o-mini"
        elif provider == "Ollama":
            self.model_name = model_name or "llama3"

    def _encode_image_base64(self, image_path):
        """Converts an image file to a base64 string."""
        with open(image_path, "rb") as image_file:
            return base64.b64encode(image_file.read()).decode("utf-8")

    def generate_text(self, prompt, system_instruction=None):
        """Generates text from a standard text prompt."""
        if self.provider == "Gemini":
            if not self.api_key:
                raise ValueError("Gemini API key is required.")
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            # Use system instructions if provided
            config = {}
            if system_instruction:
                # In google-generativeai, system instruction can be passed to GenerativeModel constructor
                model = genai.GenerativeModel(self.model_name, system_instruction=system_instruction)
            else:
                model = genai.GenerativeModel(self.model_name)
                
            response = model.generate_content(prompt)
            return response.text
            
        elif self.provider == "OpenAI":
            if not self.api_key:
                raise ValueError("OpenAI API key is required.")
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            messages = []
            if system_instruction:
                messages.append({"role": "system", "content": system_instruction})
            messages.append({"role": "user", "content": prompt})
            
            response = client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.2
            )
            return response.choices[0].message.content
            
        elif self.provider == "Ollama":
            # Call Ollama API
            url = f"{self.ollama_url}/api/generate"
            payload = {
                "model": self.model_name,
                "prompt": f"{system_instruction}\n\n{prompt}" if system_instruction else prompt,
                "stream": False,
                "options": {
                    "temperature": 0.2
                }
            }
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                return f"Error communicating with local Ollama instance: {e}. Make sure Ollama is running and the model '{self.model_name}' is downloaded."
                
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def explain_image(self, image_path, prompt):
        """Uses a Multimodal LLM to analyze an image (figure/diagram)."""
        if not os.path.exists(image_path):
            return f"Error: Image not found at {image_path}"
            
        if self.provider == "Gemini":
            if not self.api_key:
                raise ValueError("Gemini API key is required for image analysis.")
            import google.generativeai as genai
            genai.configure(api_key=self.api_key)
            
            model = genai.GenerativeModel(self.model_name)
            img = Image.open(image_path)
            
            response = model.generate_content([prompt, img])
            return response.text
            
        elif self.provider == "OpenAI":
            if not self.api_key:
                raise ValueError("OpenAI API key is required for image analysis.")
            from openai import OpenAI
            client = OpenAI(api_key=self.api_key)
            
            base64_image = self._encode_image_base64(image_path)
            
            messages = [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt},
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{base64_image}"
                            }
                        }
                    ]
                }
            ]
            
            # Ensure the selected model supports vision, fallback to gpt-4o-mini if not a standard vision-capable OpenAI model
            model_to_use = self.model_name
            if model_to_use not in ["gpt-4o", "gpt-4o-mini", "gpt-4-turbo"]:
                model_to_use = "gpt-4o-mini"
                
            response = client.chat.completions.create(
                model=model_to_use,
                messages=messages,
                max_tokens=600
            )
            return response.choices[0].message.content
            
        elif self.provider == "Ollama":
            # For Ollama, we send base64 encoded images in the generate request
            base64_image = self._encode_image_base64(image_path)
            url = f"{self.ollama_url}/api/generate"
            
            # Recommended models for Ollama vision: 'llava', 'bakllava'
            model_to_use = self.model_name
            # If they are using a standard text model, warn them or try to proceed
            if "llama" in self.model_name.lower() or "qwen" in self.model_name.lower():
                if "vl" not in self.model_name.lower() and "vision" not in self.model_name.lower():
                    # Attempting with current model, but Ollama might fail if it's not a vision model
                    pass
            
            payload = {
                "model": model_to_use,
                "prompt": prompt,
                "images": [base64_image],
                "stream": False
            }
            try:
                response = requests.post(url, json=payload, timeout=60)
                response.raise_for_status()
                return response.json().get("response", "")
            except Exception as e:
                return (f"Error communicating with local Ollama vision model: {e}. "
                        f"Make sure Ollama is running and you are using a vision-capable model like 'llava'.")
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def generate_rag_answer(self, query, contexts):
        """Generates an answer based on the provided text contexts, with citation grounding."""
        # Prepare context blocks
        context_str = ""
        for idx, ctx in enumerate(contexts):
            context_str += f"--- CONTEXT BLOCK {idx+1} (Document: {ctx['doc_name']}, Page: {ctx['page']}) ---\n"
            context_str += f"{ctx['text']}\n\n"
            
        system_instruction = (
            "You are an expert scientific research assistant. Your task is to answer questions "
            "about research papers using ONLY the provided context blocks. Keep the following rules in mind:\n"
            "1. Answer the query thoroughly, citing details directly from the text.\n"
            "2. For every claim you make that is based on a context block, append a citation showing the page number, "
            "e.g. [Page X] or (Page X) where X matches the page number in the context header. Do not invent page numbers.\n"
            "3. If the provided context blocks do not contain sufficient information to answer the question, state: "
            "'I cannot find the answer in the provided document context.' Do not make up answers.\n"
            "4. Organize your answer clearly using Markdown format (bullet points, bold text, and tables)."
        )
        
        prompt = f"Query: {query}\n\nProvided Context:\n{context_str}\nAnswer:"
        return self.generate_text(prompt, system_instruction=system_instruction)

    def generate_summary(self, doc_name, chunks):
        """Generates a structured summary of a research paper from a subset of its chunks."""
        # Typically, we feed the first few chunks (Abstract/Introduction) and some middle chunks
        # To avoid context window explosion, we select chunks representing page 1-2 and conclusion
        first_pages = [c for c in chunks if c["page"] <= 2]
        last_page_num = max(c["page"] for c in chunks) if chunks else 1
        last_pages = [c for c in chunks if c["page"] >= last_page_num - 1]
        
        # Combine selected text
        selected_chunks = []
        seen_texts = set()
        for c in (first_pages + last_pages):
            if c["text"] not in seen_texts:
                selected_chunks.append(c)
                seen_texts.add(c["text"])
                
        # If we still have too much, limit to first 12 chunks
        selected_chunks = selected_chunks[:12]
        
        context_str = ""
        for idx, c in enumerate(selected_chunks):
            context_str += f"[Page {c['page']}]: {c['text']}\n\n"
            
        system_instruction = (
            "You are a professional research evaluator. Analyze the text of the research paper provided and generate "
            "a highly structured summary. Your summary should contain the following headings:\n"
            "## 📝 Overview & Abstract Summary\n"
            "A concise 3-4 sentence paragraph summarizing the core problem and solution.\n"
            "## 💡 Key Contributions\n"
            "Bullet points highlighting the major novel contributions of this paper.\n"
            "## 🛠️ Methodology & Architecture\n"
            "Describe the datasets used, the model architecture, and the training procedure.\n"
            "## 📈 Results & Performance\n"
            "Summarize the key experimental outcomes, metrics achieved, and how they compare to baseline methods.\n"
            "## ⚠️ Limitations & Future Work\n"
            "Identify the limitations acknowledged by the authors or visible from the results, and proposed future directions."
        )
        
        prompt = f"Paper Title: {doc_name}\n\nPaper Content:\n{context_str}\n\nGenerate Summary:"
        return self.generate_text(prompt, system_instruction=system_instruction)

    def _get_representative_chunks(self, chunks, k=8):
        """Selects a representative sample of chunks from the start, middle, and end of the document."""
        if len(chunks) <= k:
            return chunks
        
        start_count = k // 2
        end_count = k - start_count - 2
        mid_count = 2
        
        start_chunks = chunks[:start_count]
        end_chunks = chunks[-end_count:]
        
        # Get middle chunks evenly spaced
        mid_chunks = []
        if len(chunks) > k:
            step = (len(chunks) - start_count - end_count) // (mid_count + 1)
            for i in range(1, mid_count + 1):
                idx = start_count + (i * step)
                if idx < len(chunks) - end_count:
                    mid_chunks.append(chunks[idx])
                    
        # Combine and sort by page number
        representative = list(set(start_chunks + mid_chunks + end_chunks))
        representative.sort(key=lambda x: x["page"])
        return representative

    def generate_comparison(self, doc1_name, doc1_chunks, doc2_name, doc2_chunks):
        """Generates a comparison between two papers."""
        # Get representative chunks (start, middle, and end) to capture abstract, results, and conclusion
        d1_rep = self._get_representative_chunks(doc1_chunks, k=8)
        d2_rep = self._get_representative_chunks(doc2_chunks, k=8)
        d1_context = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in d1_rep])
        d2_context = "\n\n".join([f"[Page {c['page']}]: {c['text']}" for c in d2_rep])
        
        system_instruction = (
            "You are a research analyst comparing scientific literature. Based on the excerpts of the two papers, "
            "provide a detailed comparative analysis. Start with a side-by-side comparison table, followed by "
            "sections analyzing commonalities, divergences, and which paper is better suited for specific use cases.\n"
            "Format the output strictly in Markdown."
        )
        
        prompt = (
            f"Paper 1: {doc1_name}\n"
            f"Paper 1 Excerpts:\n{d1_context}\n\n"
            f"Paper 2: {doc2_name}\n"
            f"Paper 2 Excerpts:\n{d2_context}\n\n"
            "Please generate:\n"
            "1. A comparative markdown table covering: Core Concept, Architecture Type, Primary Dataset, Unique Novelty, and Acknowledged Limitations.\n"
            "2. Direct comparison sections discussing: Architectural Differences, Experimental comparison, and Synthesis."
        )
        
        return self.generate_text(prompt, system_instruction=system_instruction)
