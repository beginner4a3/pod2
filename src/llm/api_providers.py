"""
API-based LLM Providers

Optional providers that require API keys:
- OpenAI (GPT-4)
- Google Gemini
"""

import os
from typing import Optional, Dict
from abc import ABC, abstractmethod


class BaseLLMProvider(ABC):
    """Abstract base class for LLM providers"""
    
    @abstractmethod
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        pass
    
    @abstractmethod
    def is_available(self) -> bool:
        pass


class OpenAIProvider(BaseLLMProvider):
    """OpenAI GPT provider (requires OPENAI_API_KEY)"""
    
    def __init__(self, model: str = "gpt-4"):
        self.model = model
        self.api_key = os.environ.get("OPENAI_API_KEY", "")
        
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_available():
            raise ValueError("OpenAI API key not set. Set OPENAI_API_KEY environment variable.")
            
        from openai import OpenAI
        
        client = OpenAI(api_key=self.api_key)
        
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        response = client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=0.7
        )
        
        return response.choices[0].message.content


class GeminiProvider(BaseLLMProvider):
    """Google Gemini provider (requires GEMINI_API_KEY)"""
    
    def __init__(self, model: str = "gemini-pro"):
        self.model = model
        self.api_key = os.environ.get("GEMINI_API_KEY", "")
        
    def is_available(self) -> bool:
        return bool(self.api_key)
    
    def generate(self, prompt: str, system_prompt: Optional[str] = None) -> str:
        if not self.is_available():
            raise ValueError("Gemini API key not set. Set GEMINI_API_KEY environment variable.")
            
        import google.generativeai as genai
        
        genai.configure(api_key=self.api_key)
        model = genai.GenerativeModel(self.model)
        
        full_prompt = prompt
        if system_prompt:
            full_prompt = f"{system_prompt}\n\n{prompt}"
            
        response = model.generate_content(full_prompt)
        return response.text


def get_available_providers() -> Dict[str, BaseLLMProvider]:
    """Get all available API providers."""
    providers = {}
    
    openai = OpenAIProvider()
    if openai.is_available():
        providers["openai"] = openai
        
    gemini = GeminiProvider()
    if gemini.is_available():
        providers["gemini"] = gemini
        
    return providers


if __name__ == "__main__":
    providers = get_available_providers()
    print(f"Available API providers: {list(providers.keys())}")
    if not providers:
        print("No API keys set. Set OPENAI_API_KEY or GEMINI_API_KEY for API-based generation.")
