import requests
import os

MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")

def ask_mistral(prompt):
    url = "https://api.mistral.ai/v1/chat/completions"

    headers = {
        "Authorization": f"Bearer {MISTRAL_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}]
    }

    res = requests.post(url, json=body, headers=headers)
    data = res.json()
    return data["choices"][0]["message"]["content"]
