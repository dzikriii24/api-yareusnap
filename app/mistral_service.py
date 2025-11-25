import requests
import os
import json

MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

def ask_mistral(prompt):
    """Send prompt to Mistral AI and get response"""
    if not MISTRAL_KEY:
        return get_fallback_response()
    
    headers = {
        "Authorization": f"Bearer {MISTRAL_KEY}",
        "Content-Type": "application/json"
    }

    body = {
        "model": "mistral-small-latest",
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.7,
        "max_tokens": 1000
    }

    try:
        res = requests.post(MISTRAL_URL, json=body, headers=headers, timeout=30)
        res.raise_for_status()
        data = res.json()
        return data["choices"][0]["message"]["content"]
    except Exception as e:
        print(f"❌ Mistral API error: {e}")
        return get_fallback_response()

def test_mistral_connection():
    """Test connection to Mistral AI"""
    if not MISTRAL_KEY:
        return "disabled", "MISTRAL_API_KEY not set"
    
    try:
        headers = {
            "Authorization": f"Bearer {MISTRAL_KEY}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Hello, reply with 'OK'"}],
            "max_tokens": 10
        }
        
        res = requests.post(MISTRAL_URL, json=body, headers=headers, timeout=10)
        res.raise_for_status()
        return "connected", "Mistral AI connection successful"
    except Exception as e:
        return "error", f"Mistral AI connection failed: {str(e)}"

def get_fallback_response():
    """Fallback response when Mistral is unavailable"""
    return json.dumps({
        "food_type": "Makanan Terdeteksi",
        "components": ["Analisis terbatas"],
        "nutrition": {
            "protein": "Perlu analisis lebih lanjut",
            "carbs": "Perlu analisis lebih lanjut", 
            "fat": "Perlu analisis lebih lanjut",
            "fiber": "Perlu analisis lebih lanjut",
            "vitamins": "Perlu analisis lebih lanjut"
        },
        "deficiencies": ["Data nutrisi terbatas"],
        "recommendations": [
            "Konsultasi dengan ahli gizi untuk analisis lengkap",
            "Variasi makanan dengan buah dan sayuran",
            "Perhatikan porsi makan yang seimbang",
            "Minum air yang cukup"
        ]
    })