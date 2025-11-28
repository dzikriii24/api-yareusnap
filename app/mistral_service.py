import requests
import os
import json
import aiohttp
import asyncio
import time

MISTRAL_KEY = os.getenv("MISTRAL_API_KEY")
MISTRAL_URL = "https://api.mistral.ai/v1/chat/completions"

# Async session for better performance
async_session = None

def get_async_session():
    """Get or create async session"""
    global async_session
    if async_session is None:
        async_session = aiohttp.ClientSession()
    return async_session

async def ask_mistral_async(prompt):
    """Async Mistral API call"""
    if not MISTRAL_KEY:
        return get_fallback_response()
    
    try:
        session = get_async_session()
        
        headers = {
            "Authorization": f"Bearer {MISTRAL_KEY}",
            "Content-Type": "application/json"
        }

        body = {
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.7,
            "max_tokens": 800,  # Reduced for speed
            "stream": False
        }

        async with session.post(
            MISTRAL_URL, 
            json=body, 
            headers=headers, 
            timeout=15  # Shorter timeout
        ) as response:
            data = await response.json()
            return data["choices"][0]["message"]["content"]
            
    except asyncio.TimeoutError:
        print("⏰ Mistral API timeout")
        return get_fallback_response()
    except Exception as e:
        print(f"❌ Mistral API error: {e}")
        return get_fallback_response()

def ask_mistral(prompt):
    """Sync version for compatibility"""
    if not MISTRAL_KEY:
        return get_fallback_response()
    
    # Run async function in event loop
    try:
        loop = asyncio.get_event_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
    
    return loop.run_until_complete(ask_mistral_async(prompt))

def test_mistral_connection():
    """Fast connection test"""
    if not MISTRAL_KEY:
        return "disabled", "MISTRAL_API_KEY not set"
    
    try:
        # Quick test dengan request kecil
        headers = {
            "Authorization": f"Bearer {MISTRAL_KEY}",
            "Content-Type": "application/json"
        }
        
        body = {
            "model": "mistral-small-latest",
            "messages": [{"role": "user", "content": "Say OK"}],
            "max_tokens": 5
        }
        
        response = requests.post(MISTRAL_URL, json=body, headers=headers, timeout=5)
        response.raise_for_status()
        return "connected", "Mistral AI connection successful"
    except Exception as e:
        return "error", f"Mistral AI connection failed: {str(e)}"

def get_fallback_response():
    """Fast fallback response"""
    return json.dumps({
        "food_type": "Makanan Terdeteksi",
        "components": ["Analisis cepat"],
        "nutrition": {
            "protein": "Cukup",
            "carbs": "Cukup", 
            "fat": "Cukup",
            "fiber": "Cukup",
            "vitamins": "Cukup"
        },
        "deficiencies": ["Perlu analisis lebih detail"],
        "recommendations": [
            "Konsumsi makanan seimbang",
            "Perbanyak buah dan sayuran",
            "Minum air yang cukup"
        ]
    })

async def close_async_session():
    """Close async session on shutdown"""
    global async_session
    if async_session:
        await async_session.close()