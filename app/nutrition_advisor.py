def build_comprehensive_prompt(food_list, detections):
    """Build comprehensive prompt for nutrition analysis"""
    
    if not food_list:
        food_list = ["Tidak ada makanan terdeteksi"]
    
    food_str = ", ".join(food_list)
    
    # Build detection details
    detection_details = ""
    if detections:
        detection_details = "\nDetail deteksi:\n"
        for det in detections[:5]:  # Limit to top 5 detections
            detection_details += f"- {det['label']} (confidence: {det['confidence']:.2f})\n"
    
    prompt = f"""
ANALISIS MAKANAN DAN NUTRISI - FORMAT JSON

Data Input:
Makanan yang terdeteksi: {food_str}
{detection_details}

Tugas Anda sebagai ahli gizi:
1. ANALISIS JENIS MAKANAN:
   - Klasifikasikan jenis makanan (berat/ringan/tradisional/modern)
   - Identifikasi komponen bahan utama

2. ANALISIS KANDUNGAN GIZI:
   - Protein: estimasi tingkat (tinggi/sedang/rendah)
   - Karbohidrat: estimasi tingkat (tinggi/sedang/rendah) 
   - Lemak: estimasi tingkat (tinggi/sedang/rendah)
   - Serat: estimasi tingkat (tinggi/sedang/rendah)
   - Vitamin & Mineral: identifikasi yang dominan

3. IDENTIFIKASI KEKURANGAN:
   - Deteksi potensi kekurangan gizi
   - Analisis ketidakseimbangan nutrisi

4. REKOMENDASI:
   - Saran makanan pendamping untuk gizi seimbang
   - Rekomendasi porsi yang tepat
   - Tips penyajian yang lebih sehat

FORMAT OUTPUT (JSON):
{{
    "food_type": "string (jenis makanan)",
    "components": ["array", "komponen", "bahan"],
    "nutrition": {{
        "protein": "string (tinggi/sedang/rendah)",
        "carbs": "string (tinggi/sedang/rendah)",
        "fat": "string (tinggi/sedang/rendah)", 
        "fiber": "string (tinggi/sedang/rendah)",
        "vitamins": "string (jenis vitamin dominan)"
    }},
    "deficiencies": ["array", "kekurangan", "gizi"],
    "recommendations": ["array", "rekomendasi", "saran"]
}}

Hanya kembalikan data JSON, tanpa penjelasan tambahan.
Analisis dalam konteks makanan Indonesia dan internasional.
"""

    return prompt

def build_simple_prompt(food_list):
    """Simple prompt for basic analysis"""
    food_str = ", ".join(food_list) if food_list else "Tidak ada makanan terdeteksi"
    
    return f"""
Analisis makanan: {food_str}

Berikan analisis singkat tentang:
1. Kandungan gizi utama
2. Kekurangan nutrisi 
3. Rekomendasi makanan pendamping

Jawab dalam bahasa Indonesia, ringkas dan praktis.
"""