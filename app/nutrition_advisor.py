def build_prompt(food_list):
    food_str = ", ".join(food_list)
    return f"""
Kamu adalah pakar nutrisi. Makanan yang terdeteksi: {food_str}.

1. Jelaskan kandungan gizi utama:
- Karbo
- Protein
- Lemak
- Serat
- Vitamin penting
2. Deteksi kekurangan gizi.
3. Berikan rekomendasi makanan tambahan untuk memenuhi gizi tersebut.
Jawab ringkas dan mudah dipahami.
"""