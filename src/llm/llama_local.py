"""
Local LLM Provider using llama-cpp-python

Supports Llama and Mistral models for self-hosted script generation.
No API keys required.
"""

import os
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass
from pathlib import Path


@dataclass
class LLMConfig:
    """Configuration for local LLM"""
    model_path: str = ""
    context_length: int = 8192
    max_tokens: int = 4096
    temperature: float = 0.7
    top_p: float = 0.9
    n_gpu_layers: int = -1  # -1 for all layers on GPU


class LlamaLocalLLM:
    """
    Local LLM using llama-cpp-python.
    
    Supports GGUF models like:
    - Mistral-7B-Instruct
    - Llama-3-8B-Instruct
    
    Usage:
        llm = LlamaLocalLLM()
        llm.load("path/to/model.gguf")
        response = llm.generate("Write a podcast script about AI")
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm = None
        
    def load(self, model_path: Optional[str] = None):
        """
        Load a GGUF model.
        
        Args:
            model_path: Path to the .gguf model file
        """
        if self.llm is not None:
            return  # Already loaded
            
        path = model_path or self.config.model_path
        if not path:
            raise ValueError("No model path provided. Please specify a GGUF model path.")
            
        if not Path(path).exists():
            raise FileNotFoundError(f"Model not found: {path}")
            
        print(f"Loading model from {path}...")
        
        from llama_cpp import Llama
        
        self.llm = Llama(
            model_path=path,
            n_ctx=self.config.context_length,
            n_gpu_layers=self.config.n_gpu_layers,
            verbose=False
        )
        
        print("Model loaded successfully!")
        
    def generate(
        self,
        prompt: str,
        system_prompt: Optional[str] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        stream: bool = False
    ) -> str:
        """
        Generate text from a prompt.
        
        Args:
            prompt: User prompt
            system_prompt: Optional system message
            max_tokens: Maximum tokens to generate
            temperature: Sampling temperature
            stream: Whether to stream the response
            
        Returns:
            Generated text
        """
        if self.llm is None:
            raise RuntimeError("Model not loaded. Call load() first.")
            
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = self.llm.create_chat_completion(
            messages=messages,
            max_tokens=max_tokens or self.config.max_tokens,
            temperature=temperature or self.config.temperature,
            top_p=self.config.top_p,
            stream=stream
        )
        
        if stream:
            return self._stream_response(response)
        else:
            return response["choices"][0]["message"]["content"]
    
    def _stream_response(self, response) -> Generator[str, None, None]:
        """Stream response tokens."""
        for chunk in response:
            delta = chunk.get("choices", [{}])[0].get("delta", {})
            content = delta.get("content", "")
            if content:
                yield content
                
    def generate_podcast_script(
        self,
        topic: str,
        language: str = "hindi",
        num_turns: int = 15,
        style: str = "conversational"
    ) -> Dict:
        """
        Generate a podcast script.
        
        Args:
            topic: Topic or content to discuss
            language: Target language
            num_turns: Number of conversation turns
            style: Conversation style
            
        Returns:
            Dictionary with script data
        """
        system_prompt = self._get_script_system_prompt(language, style)
        user_prompt = self._get_script_user_prompt(topic, num_turns, language)
        
        response = self.generate(user_prompt, system_prompt=system_prompt)
        
        # Parse JSON response
        import json
        try:
            # Try to extract JSON from response
            json_start = response.find("{")
            json_end = response.rfind("}") + 1
            if json_start >= 0 and json_end > json_start:
                return json.loads(response[json_start:json_end])
        except json.JSONDecodeError:
            pass
            
        # Fallback: Return raw response
        return {"raw_response": response}
    
    def _get_script_system_prompt(self, language: str, style: str) -> str:
        """Get system prompt for script generation."""
        return f"""You are an expert podcast script writer for Indian language podcasts.
You write engaging, natural conversations in {language}.
Your scripts are {style} in tone and perfect for text-to-speech synthesis.

Rules:
1. Write the script in {language} language
2. Output in JSON format with "title", "speakers", and "turns" fields
3. Each turn has "speaker", "text", and "emotion" fields
4. Emotions can be: happy, neutral, conversation, surprise, etc.
5. Make the conversation natural and engaging
6. Use proper punctuation for natural pauses"""
    
    def _get_script_user_prompt(self, topic: str, num_turns: int, language: str) -> str:
        """Get user prompt for script generation."""
        return f"""Write a podcast script about the following topic:

{topic}

Requirements:
- Language: {language}
- Number of turns: approximately {num_turns}
- Include proper introductions
- Make it informative yet entertaining
- End with a conclusion

Output the script in this JSON format:
{{
  "title": "Podcast Title",
  "speakers": ["Speaker1", "Speaker2"],
  "turns": [
    {{"speaker": "Speaker1", "text": "...", "emotion": "neutral"}},
    {{"speaker": "Speaker2", "text": "...", "emotion": "happy"}},
    ...
  ]
}}"""


def download_model(
    repo_id: str = "TheBloke/Mistral-7B-Instruct-v0.2-GGUF",
    filename: str = "mistral-7b-instruct-v0.2.Q4_K_M.gguf",
    cache_dir: Optional[str] = None
) -> str:
    """
    Download a model from HuggingFace Hub.
    
    Args:
        repo_id: HuggingFace repository ID
        filename: Model filename
        cache_dir: Optional cache directory
        
    Returns:
        Path to downloaded model
    """
    from huggingface_hub import hf_hub_download
    
    print(f"Downloading {filename} from {repo_id}...")
    path = hf_hub_download(
        repo_id=repo_id,
        filename=filename,
        cache_dir=cache_dir
    )
    print(f"Model downloaded to: {path}")
    return path


if __name__ == "__main__":
    # Quick test
    print("Local LLM module ready.")
    print("To use, download a GGUF model and call:")
    print("  llm = LlamaLocalLLM()")
    print("  llm.load('path/to/model.gguf')")
