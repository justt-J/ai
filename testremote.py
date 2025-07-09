import requests
import json

# Ollama API endpoint
ollama_url = "http://140.31.105.160:11434/api/generate"

# Request payload
payload = {
    "model": "mistral:7b",
    "prompt": "What is the capital of France?",
    "stream": False
}

# Headers
headers = {
    "Content-Type": "application/json"
}

# Send request and print only the response field
try:
    response = requests.post(ollama_url, headers=headers, data=json.dumps(payload))
    if response.status_code == 200:
        result = response.json()
        print(result.get("response"))  # Only print the generated response
    else:
        print(f"Error {response.status_code}: {response.text}")
except requests.exceptions.RequestException as e:
    print(f"Error connecting to Ollama server: {e}")
