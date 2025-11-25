from fastapi import FastAPI, UploadFile, File, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import uuid
from app.yolo_detector import detect_food_with_details
from app.mistral_service import ask_mistral
from app.nutrition_advisor import build_comprehensive_prompt
import json

app = FastAPI(
    title="Yareusnap AI Food Detection API",
    description="AI-powered food detection and nutrition analysis system",
    version="2.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

UPLOAD_DIR = "uploads"
os.makedirs(UPLOAD_DIR, exist_ok=True)

# Mount static files
app.mount("/static", StaticFiles(directory="static"), name="static")

@app.post("/predict")
async def predict(file: UploadFile = File(...)):
    """Endpoint utama untuk deteksi makanan dan analisis nutrisi"""
    try:
        # Validasi file
        if not file.filename:
            raise HTTPException(status_code=400, detail="No file provided")
        
        # Generate unique filename
        file_extension = os.path.splitext(file.filename)[1]
        unique_filename = f"{uuid.uuid4()}{file_extension}"
        save_path = f"{UPLOAD_DIR}/{unique_filename}"
        
        # Save uploaded file
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        print(f"📨 Received file: {file.filename} -> {unique_filename}")
        
        # YOLO detection with details
        detection_result = detect_food_with_details(save_path)
        detected_foods = detection_result["detected_foods"]
        detections = detection_result["detections"]
        
        print(f"🎯 Detected {len(detected_foods)} food items: {detected_foods}")
        
        # Nutrition analysis dengan Mistral AI
        prompt = build_comprehensive_prompt(detected_foods, detections)
        nutrition_advice = ask_mistral(prompt)
        
        # Parse Mistral response
        advice_data = parse_mistral_response(nutrition_advice)
        
        # Clean up uploaded file
        try:
            os.remove(save_path)
        except:
            pass
        
        return {
            "success": True,
            "filename": file.filename,
            "detected_foods": detected_foods,
            "detections": detections,
            "nutrition_analysis": advice_data,
            "analysis_source": "Mistral AI",
            "message": f"Successfully detected {len(detected_foods)} food items"
        }
        
    except Exception as e:
        print(f"❌ Error in prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/analyze-food")
async def analyze_food(file: UploadFile = File(...)):
    """Endpoint alternatif dengan response format yang lebih detail"""
    return await predict(file)

@app.get("/detect-only")
async def detect_only(file: UploadFile = File(...)):
    """Endpoint hanya untuk deteksi makanan tanpa analisis nutrisi"""
    try:
        # Save file
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        save_path = f"{UPLOAD_DIR}/{unique_filename}"
        
        with open(save_path, "wb") as f:
            shutil.copyfileobj(file.file, f)
        
        # YOLO detection only
        detection_result = detect_food_with_details(save_path)
        
        # Clean up
        try:
            os.remove(save_path)
        except:
            pass
        
        return {
            "success": True,
            "detection_result": detection_result
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Detection failed: {str(e)}")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    try:
        # Test model loading
        from app.yolo_detector import model
        from app.mistral_service import test_mistral_connection
        
        model_status = "healthy" if model is not None else "degraded"
        mistral_status, mistral_message = test_mistral_connection()
        
        return {
            "status": "healthy",
            "service": "Yareusnap Food Detection API",
            "model": model_status,
            "model_classes": len(model.names) if model else 0,
            "mistral_ai": mistral_status,
            "version": "2.0.0"
        }
    except Exception as e:
        return {
            "status": "degraded",
            "error": str(e)
        }

@app.get("/model-info")
async def model_info():
    """Get model information"""
    try:
        from app.yolo_detector import model
        
        return {
            "model_name": "YOLO Food Detector",
            "classes_count": len(model.names),
            "classes": list(model.names.values()),
            "input_size": "640x640",
            "confidence_threshold": 0.25
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Model info unavailable: {str(e)}")

@app.get("/test-detection")
async def test_detection():
    """Test endpoint dengan sample detection"""
    try:
        # Create a simple test case
        test_foods = ["pizza", "salad", "apple"]
        
        prompt = build_comprehensive_prompt(test_foods, [])
        nutrition_advice = ask_mistral(prompt)
        advice_data = parse_mistral_response(nutrition_advice)
        
        return {
            "success": True,
            "test_data": True,
            "detected_foods": test_foods,
            "nutrition_analysis": advice_data,
            "message": "Test detection completed successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Test failed: {str(e)}")

@app.get("/", response_class=HTMLResponse)
async def home():
    """Home page dengan interface yang lebih baik"""
    return """
    <!DOCTYPE html>
    <html>
    <head>
        <title>Yareusnap AI Food Detection</title>
        <meta charset="utf-8">
        <meta name="viewport" content="width=device-width, initial-scale=1">
        <style>
            * { margin: 0; padding: 0; box-sizing: border-box; }
            body { 
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                min-height: 100vh;
                padding: 20px;
            }
            .container {
                max-width: 1200px;
                margin: 0 auto;
                background: white;
                border-radius: 20px;
                box-shadow: 0 20px 40px rgba(0,0,0,0.1);
                overflow: hidden;
            }
            .header {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                padding: 40px;
                text-align: center;
            }
            .header h1 {
                font-size: 2.5em;
                margin-bottom: 10px;
            }
            .header p {
                font-size: 1.2em;
                opacity: 0.9;
            }
            .content {
                padding: 40px;
                display: grid;
                grid-template-columns: 1fr 1fr;
                gap: 40px;
            }
            .upload-section {
                background: #f8f9fa;
                padding: 30px;
                border-radius: 15px;
                border: 2px dashed #dee2e6;
            }
            .info-section {
                background: #f8f9fa;
                padding: 30px;
                border-radius: 15px;
            }
            .file-input {
                width: 100%;
                padding: 15px;
                margin: 20px 0;
                border: 2px solid #dee2e6;
                border-radius: 10px;
                background: white;
            }
            .btn {
                background: linear-gradient(135deg, #4facfe 0%, #00f2fe 100%);
                color: white;
                border: none;
                padding: 15px 30px;
                border-radius: 10px;
                font-size: 1.1em;
                cursor: pointer;
                width: 100%;
                transition: transform 0.2s;
            }
            .btn:hover {
                transform: translateY(-2px);
                box-shadow: 0 10px 20px rgba(0,0,0,0.2);
            }
            .result {
                margin-top: 20px;
                padding: 20px;
                background: white;
                border-radius: 10px;
                border-left: 5px solid #4facfe;
                display: none;
            }
            .endpoint {
                background: white;
                padding: 15px;
                margin: 10px 0;
                border-radius: 8px;
                border-left: 4px solid #28a745;
            }
            .endpoint h4 {
                color: #495057;
                margin-bottom: 5px;
            }
            .endpoint code {
                background: #e9ecef;
                padding: 2px 6px;
                border-radius: 4px;
                font-family: monospace;
            }
            @media (max-width: 768px) {
                .content { grid-template-columns: 1fr; }
                .header h1 { font-size: 2em; }
            }
        </style>
    </head>
    <body>
        <div class="container">
            <div class="header">
                <h1>🍽️ Yareusnap AI Food Detection</h1>
                <p>AI-powered food recognition and nutrition analysis</p>
            </div>
            
            <div class="content">
                <div class="upload-section">
                    <h2>📤 Upload Food Image</h2>
                    <p>Upload gambar makanan untuk deteksi dan analisis gizi</p>
                    
                    <input type="file" id="fileInput" class="file-input" accept="image/*">
                    <button class="btn" onclick="analyzeFood()">🔍 Analyze Food</button>
                    
                    <div id="result" class="result">
                        <h3>📊 Analysis Result</h3>
                        <div id="resultContent"></div>
                    </div>
                </div>
                
                <div class="info-section">
                    <h2>🔧 API Endpoints</h2>
                    <p>Available endpoints for integration:</p>
                    
                    <div class="endpoint">
                        <h4>POST /predict</h4>
                        <p>Deteksi makanan + analisis nutrisi lengkap</p>
                        <code>Content-Type: multipart/form-data</code>
                    </div>
                    
                    <div class="endpoint">
                        <h4>POST /analyze-food</h4>
                        <p>Alternatif endpoint dengan response detail</p>
                        <code>Content-Type: multipart/form-data</code>
                    </div>
                    
                    <div class="endpoint">
                        <h4>GET /health</h4>
                        <p>Health check dan status sistem</p>
                        <code>No authentication required</code>
                    </div>
                    
                    <div class="endpoint">
                        <h4>GET /model-info</h4>
                        <p>Informasi model dan classes yang tersedia</p>
                    </div>
                    
                    <h3 style="margin-top: 30px;">📋 Features</h3>
                    <ul style="margin-left: 20px; margin-top: 10px;">
                        <li>Deteksi 100+ jenis makanan</li>
                        <li>Analisis nutrisi dengan Mistral AI</li>
                        <li>Rekomendasi kesehatan</li>
                        <li>RESTful API</li>
                        <li>CORS enabled</li>
                    </ul>
                </div>
            </div>
        </div>

        <script>
            async function analyzeFood() {
                const fileInput = document.getElementById('fileInput');
                const resultDiv = document.getElementById('result');
                const resultContent = document.getElementById('resultContent');
                
                if (!fileInput.files[0]) {
                    alert('Please select an image file');
                    return;
                }
                
                const file = fileInput.files[0];
                const formData = new FormData();
                formData.append('file', file);
                
                resultContent.innerHTML = '<p>🔄 Analyzing food... Please wait.</p>';
                resultDiv.style.display = 'block';
                
                try {
                    const response = await fetch('/predict', {
                        method: 'POST',
                        body: formData
                    });
                    
                    const data = await response.json();
                    
                    if (data.success) {
                        let html = `
                            <p><strong>✅ Detected Foods:</strong> ${data.detected_foods.join(', ') || 'None'}</p>
                            <p><strong>🤖 Analysis Source:</strong> ${data.analysis_source}</p>
                            <div style="margin-top: 15px; padding: 15px; background: #f8f9fa; border-radius: 8px;">
                                <h4>📊 Nutrition Analysis:</h4>
                                <p><strong>Food Type:</strong> ${data.nutrition_analysis.food_type || 'N/A'}</p>
                                <p><strong>Components:</strong> ${(data.nutrition_analysis.components || []).join(', ') || 'N/A'}</p>
                                <p><strong>Recommendations:</strong></p>
                                <ul>
                                    ${(data.nutrition_analysis.recommendations || []).map(rec => `<li>${rec}</li>`).join('')}
                                </ul>
                            </div>
                        `;
                        resultContent.innerHTML = html;
                    } else {
                        resultContent.innerHTML = `<p style="color: red;">❌ Error: ${data.message || 'Unknown error'}</p>`;
                    }
                } catch (error) {
                    resultContent.innerHTML = `<p style="color: red;">❌ Network error: ${error.message}</p>`;
                }
            }
            
            // Drag and drop support
            const fileInput = document.getElementById('fileInput');
            const uploadSection = document.querySelector('.upload-section');
            
            uploadSection.addEventListener('dragover', (e) => {
                e.preventDefault();
                uploadSection.style.background = '#e3f2fd';
            });
            
            uploadSection.addEventListener('dragleave', (e) => {
                e.preventDefault();
                uploadSection.style.background = '#f8f9fa';
            });
            
            uploadSection.addEventListener('drop', (e) => {
                e.preventDefault();
                uploadSection.style.background = '#f8f9fa';
                fileInput.files = e.dataTransfer.files;
            });
        </script>
    </body>
    </html>
    """

def parse_mistral_response(response_text):
    """Parse Mistral AI response menjadi structured data"""
    try:
        # Coba parse sebagai JSON pertama
        if response_text.strip().startswith('{'):
            return json.loads(response_text)
    except:
        pass
    
    # Fallback: extract information dari text response
    lines = response_text.split('\n')
    analysis = {
        "food_type": "Makanan Terdeteksi",
        "components": [],
        "nutrition": {
            "protein": "Tidak Diketahui",
            "carbs": "Tidak Diketahui", 
            "fat": "Tidak Diketahui",
            "fiber": "Tidak Diketahui",
            "vitamins": "Tidak Diketahui"
        },
        "deficiencies": [],
        "recommendations": []
    }
    
    current_section = None
    for line in lines:
        line = line.strip()
        if not line:
            continue
            
        # Detect sections
        if 'komponen' in line.lower() or 'bahan' in line.lower():
            current_section = 'components'
        elif 'gizi' in line.lower() or 'nutrisi' in line.lower():
            current_section = 'nutrition'
        elif 'kekurangan' in line.lower():
            current_section = 'deficiencies'
        elif 'rekomendasi' in line.lower() or 'saran' in line.lower():
            current_section = 'recommendations'
        elif line.startswith('-') or line.startswith('•'):
            item = line[1:].strip()
            if current_section == 'components' and item:
                analysis['components'].append(item)
            elif current_section == 'deficiencies' and item:
                analysis['deficiencies'].append(item)
            elif current_section == 'recommendations' and item:
                analysis['recommendations'].append(item)
    
    # Jika tidak ada recommendations, tambahkan default
    if not analysis['recommendations']:
        analysis['recommendations'] = [
            "Konsultasi dengan ahli gizi untuk analisis lengkap",
            "Perhatikan porsi dan variasi makanan",
            "Minum air yang cukup sepanjang hari"
        ]
    
    return analysis

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)