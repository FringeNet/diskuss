# Module for integrating with Ollama 

import requests
import json

OLLAMA_API_BASE_URL = "http://localhost:11434"

def is_ollama_running():
    """Checks if the Ollama service is running."""
    try:
        response = requests.get(f"{OLLAMA_API_BASE_URL}/", timeout=5)
        return response.status_code == 200
    except requests.exceptions.RequestException:
        return False

def list_ollama_models():
    """Lists the available models in Ollama."""
    if not is_ollama_running():
        return None
    try:
        response = requests.get(f"{OLLAMA_API_BASE_URL}/api/tags")
        response.raise_for_status()
        return [model['name'] for model in response.json().get('models', [])]
    except requests.exceptions.RequestException:
        return None

def generate_embedding(model_name: str, text: str):
    """Generates an embedding for the given text using the specified Ollama model."""
    if not is_ollama_running():
        return None
    try:
        payload = {
            "model": model_name,
            "prompt": text,
            "stream": False
        }
        response = requests.post(f"{OLLAMA_API_BASE_URL}/api/embeddings", json=payload)
        response.raise_for_status()
        return response.json().get('embedding')
    except requests.exceptions.RequestException:
        return None

def chat_with_ollama(model_name: str, messages: list):
    """Interacts with the Ollama chat model."""
    if not is_ollama_running():
        return None
    try:
        payload = {
            "model": model_name,
            "messages": messages,
            "stream": False
        }
        response = requests.post(f"{OLLAMA_API_BASE_URL}/api/chat", json=payload)
        response.raise_for_status()
        return response.json().get('message', {}).get('content')
    except requests.exceptions.RequestException:
        return None

if __name__ == '__main__':
    # Example usage (for testing)
    print(f"Ollama running: {is_ollama_running()}")
    models = list_ollama_models()
    print(f"Available models: {models}")

    if models:
        # Assuming a model like 'nomic-embed-text' is available for embeddings
        embedding_model = next((m for m in models if 'embed' in m), None)
        if embedding_model:
            text_to_embed = "This is a test sentence."
            embedding = generate_embedding(embedding_model, text_to_embed)
            print(f"Embedding for '{text_to_embed[:20]}...': {embedding[:10]}..." if embedding else "Embedding generation failed.")
        else:
            print("No embedding model found.")

        # Assuming a chat model like 'llama3' is available
        chat_model = next((m for m in models if 'llama' in m), None) # Simple check for a llama model
        if chat_model:
            messages = [{"role": "user", "content": "Tell me a short story."}]
            response = chat_with_ollama(chat_model, messages)
            print(f"Chat response: {response[:50]}..." if response else "Chat interaction failed.")
        else:
            print("No chat model found.")