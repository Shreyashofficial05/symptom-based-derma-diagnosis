from flask import Flask, request, jsonify
from flask_cors import CORS
import os
import base64
import io
from PIL import Image
import sys
import torch

MODEL_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../models"))
if MODEL_DIR not in sys.path:
    sys.path.append(MODEL_DIR)

from predict import load_model, process_image, predict
from text_classifier import disease_to_symptoms, get_unique_symptoms
from disease_mappings import CLASS_NAMES, name_mapping, get_display_name
from ensemble import run_ensemble_model

app = Flask(__name__)
CORS(app)


MODEL_PATH = os.path.join(os.path.dirname(__file__), "../../models/image_model_1.pth")
model, device, num_classes = load_model(MODEL_PATH)

def process_base64_image(base64_string):
    # Remove data URL prefix if present
    if "," in base64_string:
        base64_string = base64_string.split(",")[1]
    
    # Decode base64 string to bytes
    image_bytes = base64.b64decode(base64_string)
    
    # Convert to PIL Image
    image = Image.open(io.BytesIO(image_bytes))
    return image

@app.route("/predict", methods=["POST"])
def predict_image():
    try:
        # Get base64 image from request
        data = request.get_json()
        if not data or "image" not in data:
            return jsonify({"error": "No image provided"}), 400

        # Process the base64 image
        image = process_base64_image(data["image"])
        
        # Save image to temp file
        temp_path = "temp_image.jpg"
        image.save(temp_path)
        
        # Process image for model
        image_tensor = process_image(temp_path)
        
        # Get prediction
        pred_indices, probabilities = predict(model, image_tensor, device)
        
        # Format results and get relevant symptoms
        results = []
        for idx, prob in zip(pred_indices, probabilities):
            # Get internal model name
            class_name = CLASS_NAMES[idx] if idx < len(CLASS_NAMES) else f"Class_{idx}"
            
            # Get display name using the same mapping as ensemble.py
            mapped_name = get_display_name(class_name)
            
            # Get symptoms for this disease
            symptoms = disease_to_symptoms.get(mapped_name, [])
            
            results.append({
                "disease": mapped_name,  # Use mapped name consistently
                "confidence": float(prob) * 100,
                "symptoms": symptoms
            })
        
        # Get unique symptoms from top 5 predictions only
        top_symptoms = []
        for result in results[:5]:  # Only top 5 predictions
            top_symptoms.extend(result["symptoms"])
        top_symptoms = list(set(top_symptoms))  # Remove duplicates
        
        # Clean up temp file
        os.remove(temp_path)
        
        return jsonify({
            "predictions": results,
            "topSymptoms": top_symptoms
        })
    
    except Exception as e:
        return jsonify({"error": str(e)}), 500

# Endpoint to fetch unique symptoms
@app.route("/unique-symptoms", methods=["GET"])
def unique_symptoms():
    symptoms = get_unique_symptoms()
    return jsonify({"unique_symptoms": symptoms})

# Endpoint to run ensemble model
@app.route("/run-ensemble", methods=["POST"])
def run_ensemble():
    try:
        data = request.get_json()
        if not data or "selected_symptoms" not in data:
            return jsonify({"error": "No symptoms provided"}), 400

        selected_symptoms = data["selected_symptoms"]
        image_path = None
        w_img = 0.5  # default weight
        
        # Handle image data and weights
        if "image" in data and data["image"]:
            # Save base64 image to temporary file
            image = process_base64_image(data["image"])
            image_path = "temp_ensemble_image.jpg"
            image.save(image_path)
            
        if "weight_image" in data:
            w_img = float(data["weight_image"])

        try:
            results = run_ensemble_model(selected_symptoms, image_path, w_img)
            return jsonify({
                "predictions": results,
                "weights": {
                    "image": w_img,
                    "symptoms": 1.0 - w_img
                }
            })
        finally:
            # Clean up temporary image file if it exists
            if image_path and os.path.exists(image_path):
                os.remove(image_path)

    except Exception as e:
        return jsonify({"error": str(e)}), 500

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5002)
