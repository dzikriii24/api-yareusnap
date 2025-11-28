def build_comprehensive_prompt(food_list, detections):
    """Optimized prompt untuk response cepat"""
    
    if not food_list:
        food_list = ["Tidak ada makanan terdeteksi"]
    
    food_str = ", ".join(food_list)
    
    # Build detection details cepat
    detection_details = ""
    if detections and len(detections) > 0:
        detection_details = "\nDeteksi: "
        for det in detections[:3]:  # Hanya 3 teratas
            detection_details += f"{det['label']}({det['confidence']:.1f}), "
    
    prompt = f"""
ANALISIS CEPAT - FORMAT JSON

Makanan: {food_str}
{detection_details}

Analisis cepat:
- Jenis makanan
- Komponen utama  
- Estimasi gizi (protein, karbo, lemak, serat, vitamin)
- 2 rekomendasi utama

OUTPUT JSON:
{{
    "food_type": "string",
    "components": ["item1", "item2"],
    "nutrition": {{
        "protein": "tinggi/sedang/rendah",
        "carbs": "tinggi/sedang/rendah", 
        "fat": "tinggi/sedang/rendah",
        "fiber": "tinggi/sedang/rendah",
        "vitamins": "jenis vitamin"
    }},
    "deficiencies": ["kekurangan1", "kekurangan2"],
    "recommendations": ["rekom1", "rekom2"]
}}

Hanya JSON, tanpa penjelasan.
"""

    return prompt

def build_simple_prompt(food_list):
    """Super simple prompt untuk kecepatan maksimal"""
    food_str = ", ".join(food_list) if food_list else "Tidak terdeteksi"
    
    return f"""
Makanan: {food_str}
Analisis singkat: jenis, gizi utama, 1 saran.
Format JSON singkat.
"""