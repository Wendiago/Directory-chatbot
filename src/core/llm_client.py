import requests
from typing import Dict, List, Generator, Union
from config.settings import Settings
from config.model_config import ModelConfig
import logging
import ollama

logger = logging.getLogger(__name__)

class LLMClient:
    def __init__(self, host: str = None, model: str = None):
        self.host = host or Settings.OLLAMA_HOST
        self.model = model or Settings.OLLAMA_MODEL
        self.session = requests.Session()
        try:
            self.client = ollama.Client(host=self.host)
            # A quick check to see if the host is reachable
            self.client.list() 
        except Exception as e:
            logger.error(f"Failed to connect to Ollama at {self.host}. Is Ollama running?")
            logger.error(f"Error: {e}")
            raise ConnectionError(f"Could not connect to Ollama at {self.host}")
    
    def pull_model(self):
        """Pulls the model from the Ollama registry if not present."""
        try:
            logger.info(f"Checking for model '{self.model}'...")
            models = self.client.list().get('models', [])
            #print("Models: ", models)
            if not any(m.model.startswith(self.model) for m in models):
                logger.info(f"Model '{self.model}' not found locally. Pulling from registry...")
                self.client.pull(self.model)
                logger.info(f"Successfully pulled model '{self.model}'.")
            else:
                logger.info(f"Model '{self.model}' is already available.")
        except Exception as e:
            logger.error(f"Error pulling model '{self.model}': {e}")
            raise
     
    def generate(self, prompt: str, system_prompt: str = None, 
                stream: bool = False) -> Union[Dict, Generator[str, None, None]]:
        """Generate response from Ollama"""
        try:
            if stream:
                response = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options=ModelConfig.OLLAMA_PARAMS,
                    system=system_prompt,
                    stream=True,
                    think=True
                )
                def stream_output():
                    printed_thinking_label = False
                    printed_response_label = False

                    for chunk in response:
                        if chunk.thinking and not printed_response_label:
                            if not printed_thinking_label:
                                printed_thinking_label = True
                                yield "Thinking: "
                            yield chunk.thinking

                        elif chunk.response:
                            if not printed_response_label:
                                printed_response_label = True
                                yield "\nResponse: "
                            yield chunk.response

                return stream_output()

            else:
                result = self.client.generate(
                    model=self.model,
                    prompt=prompt,
                    options=ModelConfig.OLLAMA_PARAMS,
                    system=system_prompt,
                    think=True
                )
                return result

        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return {"error": str(e)}
    
    def chat(self, messages: List[Dict[str, str]], stream: bool = False) -> Union[Dict, Generator[str, None, None]]:
        """Chat with the model using conversation history."""
        try:
            if stream:
                response: ollama.ChatResponse = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options=ModelConfig.OLLAMA_PARAMS,
                    stream=True,
                    think=True
                )
                def stream_output():
                    printed_thinking_label = False
                    printed_response_label = False

                    for chunk in response:
                        msg = chunk.message

                        if msg.thinking and not printed_response_label:
                            if not printed_thinking_label:
                                printed_thinking_label = True
                                yield "Thinking: "
                            yield msg.thinking

                        elif msg.content:
                            if not printed_response_label:
                                printed_response_label = True
                                yield "\nResponse: "
                            yield msg.content

                return stream_output()

            else:
                result = self.client.chat(
                    model=self.model,
                    messages=messages,
                    options=ModelConfig.OLLAMA_PARAMS
                )
                return result

        except Exception as e:
            logger.error(f"Error in chat: {e}")
            return {"error": str(e)}
