import tiktoken
from config.settings import Settings
from config.model_config import ModelConfig
from typing import List, Dict

class Utils:        
    @staticmethod
    def count_tokens(text: str, model: str = Settings.OLLAMA_MODEL) -> int:
        encoding = tiktoken.get_encoding("cl100k_base")
        return len(encoding.encode(text))
    
    @staticmethod
    def trim_history_to_fit_token_limit(history: List[Dict[str, str]], max_tokens: int |None = ModelConfig.OLLAMA_PARAMS.num_ctx, model: str = Settings.OLLAMA_MODEL) -> List[Dict[str, str]]:
        total_tokens = 0
        trimmed = []

        # Start from the end (most recent messages)
        for message in reversed(history):
            message_tokens = Utils.count_tokens(message["content"], model=model)
            if max_tokens is not None and total_tokens + message_tokens > max_tokens:
                break
            trimmed.insert(0, message)  # Add to front
            total_tokens += message_tokens

        return trimmed