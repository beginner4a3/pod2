"""
Local LLM Provider using llama-cpp-python

Supports BharatGPT, Llama and Mistral models for self-hosted script generation.
Optimized for Indian languages with BharatGPT-3B-Indic.
No API keys required.

Model should be pre-installed via: python setup_models.py
"""

import os
from typing import Optional, List, Dict, Generator
from dataclasses import dataclass
from pathlib import Path


def find_gguf_models() -> List[str]:
    """Find all GGUF model files in common locations."""
    search_paths = [
        Path.cwd(),
        Path.cwd() / "models",
        Path.home() / ".cache" / "huggingface" / "hub",
        Path("/content/pod2") if Path("/content").exists() else None,  # Colab
        Path("/content/pod2/models") if Path("/content").exists() else None,
    ]
    
    models = []
    for base_path in search_paths:
        if base_path is None or not base_path.exists():
            continue
        for gguf_file in base_path.rglob("*.gguf"):
            models.append(str(gguf_file))
    
    return models


def get_default_model_path() -> Optional[str]:
    """Get the path to a pre-installed GGUF model."""
    models = find_gguf_models()
    if models:
        # Prefer BharatGPT, mistral or llama models
        for m in models:
            if "bharat" in m.lower() or "indic" in m.lower():
                return m
        for m in models:
            if "mistral" in m.lower() or "llama" in m.lower():
                return m
        return models[0]
    
    return None


@dataclass
class LLMConfig:
    """Configuration for local LLM"""
    model_path: str = ""
    context_length: int = 4096  # Reduced for faster loading
    max_tokens: int = 1024      # Reduced for faster generation
    temperature: float = 0.7
    top_p: float = 0.9
    repeat_penalty: float = 1.2  # Prevent repetition (1.0 = disabled)
    n_gpu_layers: int = -1      # -1 = all layers on GPU


class LlamaLocalLLM:
    """
    Local LLM using llama-cpp-python.
    
    Supports GGUF models like:
    - BharatGPT-3B-Indic (Best for Indian languages)
    - Mistral-7B-Instruct
    - Llama-3-8B-Instruct
    
    Usage:
        llm = LlamaLocalLLM()
        llm.load()  # Auto-detects pre-installed model
        response = llm.generate("Write a podcast script about AI")
    """
    
    def __init__(self, config: Optional[LLMConfig] = None):
        self.config = config or LLMConfig()
        self.llm = None
        
    def load(self, model_path: Optional[str] = None):
        """
        Load a GGUF model.
        
        Args:
            model_path: Path to the .gguf model file (auto-detects if not provided)
        """
        if self.llm is not None:
            return  # Already loaded
        
        # Try to find model
        path = model_path or self.config.model_path
        
        if not path:
            # Auto-detect pre-installed model
            path = get_default_model_path()
        
        if not path:
            raise FileNotFoundError(
                "No LLM model found. Please run: python setup_models.py"
            )
            
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
            repeat_penalty=self.config.repeat_penalty,
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
        # Use fewer tokens for podcast scripts (they don't need to be very long)
        response = self.generate(user_prompt, system_prompt=system_prompt, max_tokens=800)
        
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
        # All 21 languages supported by Indic-ParlerTTS
        lang_config = {
            # North Indian Languages
            "hindi": {"greeting": "नमस्ते", "script": "Devanagari", "mix": "Hinglish"},
            "marathi": {"greeting": "नमस्कार", "script": "Devanagari", "mix": "Marathi-English"},
            "gujarati": {"greeting": "નમસ્તે", "script": "Gujarati", "mix": "Gujarati-English"},
            "punjabi": {"greeting": "ਸਤ ਸ੍ਰੀ ਅਕਾਲ", "script": "Gurmukhi", "mix": "Punjabi-English"},
            "dogri": {"greeting": "सत श्री अकाल", "script": "Devanagari", "mix": "Dogri-English"},
            "nepali": {"greeting": "नमस्कार", "script": "Devanagari", "mix": "Nepali-English"},
            "sanskrit": {"greeting": "नमस्ते", "script": "Devanagari", "mix": "Sanskrit-English"},
            
            # South Indian Languages
            "telugu": {"greeting": "నమస్కారం", "script": "Telugu", "mix": "Tenglish"},
            "tamil": {"greeting": "வணக்கம்", "script": "Tamil", "mix": "Tanglish"},
            "malayalam": {"greeting": "നമസ്കാരം", "script": "Malayalam", "mix": "Manglish"},
            "kannada": {"greeting": "ನಮಸ್ಕಾರ", "script": "Kannada", "mix": "Kannada-English"},
            
            # East Indian Languages
            "bengali": {"greeting": "নমস্কার", "script": "Bengali", "mix": "Benglish"},
            "odia": {"greeting": "ନମସ୍କାର", "script": "Odia", "mix": "Odia-English"},
            "assamese": {"greeting": "নমস্কাৰ", "script": "Assamese", "mix": "Assamese-English"},
            "manipuri": {"greeting": "ꯈꯨꯔꯨꯝꯖꯔꯤ", "script": "Meitei", "mix": "Manipuri-English"},
            "bodo": {"greeting": "नमस्कार", "script": "Devanagari", "mix": "Bodo-English"},
            
            # English
            "english": {"greeting": "Hello", "script": "Latin", "mix": "Indian English"},
        }
        
        config = lang_config.get(language.lower(), lang_config["hindi"])
        
        return f"""You are a podcast script writer for {language} language.

STRICT OUTPUT FORMAT (no other text):
Speaker1: {config['greeting']}! [dialogue in {language}]
Speaker2: [response in {language}]
Speaker1: [continue]
Speaker2: [reply]
...

RULES:
- ONLY use "Speaker1:" and "Speaker2:" prefixes
- Write in {language} script ({config['script']})
- {config['mix']} (mixing English words) is allowed
- Each line = one speaker turn
- Natural, conversational style
- NO JSON, NO brackets, NO titles, NO numbering
- ONLY Speaker1/Speaker2 dialogue lines
- Write correctly spelled words for proper TTS pronunciation"""
    
    def _get_script_user_prompt(self, topic: str, num_turns: int, language: str) -> str:
        """Get user prompt for script generation."""
        # Examples for all languages
        examples = {
            "hindi": """Speaker1: नमस्ते दोस्तों! आज हम एक interesting topic पर बात करेंगे।
Speaker2: हाँ, यह topic बहुत important है। चलिए शुरू करते हैं।
Speaker1: तो पहली बात यह है कि...
Speaker2: बिल्कुल सही! और इसके साथ...""",

            "telugu": """Speaker1: నమస్కారం! ఈ రోజు మనం ఒక important topic గురించి మాట్లాడుదాం.
Speaker2: అవును, ఇది చాలా interesting. మొదలు పెడదాం.
Speaker1: మొదటి విషయం ఏమిటంటే...
Speaker2: అవును, సరిగ్గా చెప్పారు!""",

            "tamil": """Speaker1: வணக்கம்! இன்று நாம் ஒரு important topic பற்றி பேசுவோம்.
Speaker2: ஆமா, இது மிகவும் interesting. ஆரம்பிக்கலாம்.
Speaker1: முதலில் சொல்ல வேண்டியது என்னவென்றால்...
Speaker2: சரியாக சொன்னீர்கள்!""",

            "malayalam": """Speaker1: നമസ്കാരം! ഇന്ന് നമ്മൾ ഒരു important topic കുറിച്ച് സംസാരിക്കാം.
Speaker2: അതെ, ഇത് വളരെ interesting ആണ്. തുടങ്ങാം.
Speaker1: ആദ്യം പറയേണ്ടത് എന്തെന്നാൽ...
Speaker2: തീർച്ചയായും ശരിയാണ്!""",

            "kannada": """Speaker1: ನಮಸ್ಕಾರ! ಇವತ್ತು ನಾವು ಒಂದು important topic ಬಗ್ಗೆ ಮಾತಾಡೋಣ.
Speaker2: ಹೌದು, ಇದು ತುಂಬಾ interesting. ಶುರು ಮಾಡೋಣ.
Speaker1: ಮೊದಲು ಹೇಳಬೇಕಾದದ್ದು ಏನೆಂದರೆ...
Speaker2: ಸರಿಯಾಗಿ ಹೇಳಿದ್ರಿ!""",

            "bengali": """Speaker1: নমস্কার! আজ আমরা একটা important topic নিয়ে কথা বলব।
Speaker2: হ্যাঁ, এটা খুবই interesting। শুরু করা যাক।
Speaker1: প্রথমে বলতে হয় যে...
Speaker2: একদম ঠিক বলেছেন!""",

            "marathi": """Speaker1: नमस्कार! आज आपण एक important topic वर बोलणार आहोत.
Speaker2: हो, हे खूप interesting आहे. सुरू करूया.
Speaker1: पहिली गोष्ट अशी की...
Speaker2: अगदी बरोबर!""",

            "gujarati": """Speaker1: નમસ્તે! આજે આપણે એક important topic વિશે વાત કરીશું.
Speaker2: હા, આ ખૂબ જ interesting છે. શરૂ કરીએ.
Speaker1: પહેલી વાત એ છે કે...
Speaker2: બિલકુલ સાચું!""",

            "punjabi": """Speaker1: ਸਤ ਸ੍ਰੀ ਅਕਾਲ! ਅੱਜ ਅਸੀਂ ਇੱਕ important topic ਬਾਰੇ ਗੱਲ ਕਰਾਂਗੇ.
Speaker2: ਹਾਂ ਜੀ, ਇਹ ਬਹੁਤ interesting ਹੈ. ਸ਼ੁਰੂ ਕਰੀਏ.
Speaker1: ਪਹਿਲੀ ਗੱਲ ਇਹ ਹੈ ਕਿ...
Speaker2: ਬਿਲਕੁਲ ਸਹੀ!""",

            "odia": """Speaker1: ନମସ୍କାର! ଆଜି ଆମେ ଗୋଟିଏ important topic ବିଷୟରେ କଥା ହେବା.
Speaker2: ହଁ, ଏହା ବହୁତ interesting. ଆରମ୍ଭ କରିବା.
Speaker1: ପ୍ରଥମ କଥା ହେଉଛି...
Speaker2: ସମ୍ପୂର୍ଣ୍ଣ ଠିକ୍!""",

            "assamese": """Speaker1: নমস্কাৰ! আজি আমি এটা important topic লৈ কথা পাতিম।
Speaker2: হয়, এইটো বহুত interesting। আৰম্ভ কৰোঁ।
Speaker1: প্ৰথম কথা হ'ল...
Speaker2: সম্পূৰ্ণ শুদ্ধ!""",

            "english": """Speaker1: Hello everyone! Today we'll discuss an important topic.
Speaker2: Yes, this is very interesting. Let's begin.
Speaker1: The first point is that...
Speaker2: Absolutely right!""",

            "nepali": """Speaker1: नमस्कार! आज हामी एउटा important topic बारेमा कुरा गर्नेछौं।
Speaker2: हो, यो धेरै interesting छ। सुरु गरौं।
Speaker1: पहिलो कुरा के हो भने...
Speaker2: एकदम सही!""",

            "sanskrit": """Speaker1: नमस्ते! अद्य वयं एकं महत्त्वपूर्णं विषयं चर्चयामः।
Speaker2: आम्, एतत् अतीव रोचकम् अस्ति। आरभामहे।
Speaker1: प्रथमं वक्तव्यम् अस्ति यत्...
Speaker2: सम्यक् उक्तम्!""",

            "manipuri": """Speaker1: খুরুমজরি! ঙসি অদোম্না ময়াম টপিক অমা হায়বা পাংথোক্কনি।
Speaker2: হোই, অসি য়াম্না ইন্তরেষ্টিং লৈ। হৌরক্কদবনি।
Speaker1: অহানবা পাউ অসিনা...
Speaker2: করিগুম্বা ফজবা!""",

            "bodo": """Speaker1: नमस्कार! दिनै आंनि गोमोलांखौ फोरोंनाय बिसायखौ रायज्लायो।
Speaker2: हय, बेनि बांसिन इन्टरेस्टिं। जागायनाय।
Speaker1: गिबि बिसायआ बादि...
Speaker2: गोजौनि सैथो!""",

            "dogri": """Speaker1: सत श्री अकाल! अज्ज असां इक important topic बारे गल्ल करांगे।
Speaker2: हाँ जी, एह बड़ी interesting ऐ। शुरू करदे आं।
Speaker1: पैहली गल्ल एह ऐ कि...
Speaker2: बिलकुल सई!""",
        }
        
        example = examples.get(language.lower(), examples["hindi"])
        
        return f"""Write a podcast script about:

{topic}

REQUIREMENTS:
- Language: {language}
- Turns: {num_turns} (alternating Speaker1 and Speaker2)
- Start with greeting
- End with conclusion
- Use correct spelling for proper TTS pronunciation

EXAMPLE FORMAT:
{example}

NOW WRITE THE FULL {num_turns}-TURN SCRIPT:"""


def download_model(
    repo_id: str = "QuantFactory/BharatGPT-3B-Indic-GGUF",
    filename: str = "BharatGPT-3B-Indic.Q4_K_M.gguf",
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
