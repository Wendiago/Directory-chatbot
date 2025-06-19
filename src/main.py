from core.llm_client import LLMClient
from config.model_config import ModelConfig
import logging
from utils.utils import Utils

logger = logging.getLogger(__name__)
def main():
    try: 
        client = LLMClient()
        client.pull_model() 
    except ConnectionError as e:
        logger.error(f"Could not run tests: {e}")
        return

    # prompt = "What is LLM (Large language model)?"

    # print("\n" + "="*50)
    # print("🚀 2. TESTING STREAMING GENERATE RESPONSE 🚀")
    # print("="*50)
    # print(f"Prompt: {prompt}\n")
    # print("Streaming Content:")
    # try:
    #     for chunk in client.generate(prompt=prompt, stream=True):
    #         print(chunk, end="", flush=True)
    #     print("\n--- End of Stream ---")
    # except Exception as e:
    #     print(f"\nAn error occurred during streaming: {e}")
    
    
    ######## CHAT SIMULATE ########
    conversation_history = [
        {"role": "system", "content": "You are a helpful assistant."}
    ]

    while True:
        user_input = input("You: ")
        if user_input.lower() in ("exit", "quit"):
            print("Goodbye!")
            break

        conversation_history.append({"role": "user", "content": user_input})
        trimmed_history = Utils.trim_history_to_fit_token_limit(conversation_history)

        response_stream = client.chat(messages=trimmed_history, stream=True)

        print("Bot:", end=" ", flush=True)
        try:
            response_chunks = []
            for chunk in response_stream:
                print(chunk, end="", flush=True)
                response_chunks.append(chunk)
        except KeyboardInterrupt:
            print("\n[Stream interrupted]")

        full_response = "".join(response_chunks)
        conversation_history.append({"role": "assistant", "content": full_response})
        print()

if __name__ == "__main__":
    main()