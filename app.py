"""
Plant Disease Detection Web Application

This is the main Flask application file that handles routes and serves the web interface.
"""

import os
import sys
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms, models
from PIL import Image
from flask import Flask, render_template, request, jsonify, redirect, url_for, send_from_directory
from werkzeug.utils import secure_filename
from datetime import datetime
import json
import uuid

# Add a context processor for all templates
from flask import g

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'static/uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max upload size
app.config['ALLOWED_EXTENSIONS'] = {'png', 'jpg', 'jpeg'}

# Ensure upload directory exists
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Global variables
model = None
class_indices = None
idx_to_class = None
device = None
has_loaded_model = False

def allowed_file(filename):
    """Check if the uploaded file has an allowed extension."""
    return '.' in filename and \
           filename.rsplit('.', 1)[1].lower() in app.config['ALLOWED_EXTENSIONS']

def get_device():
    """Determine the best available device for model inference."""
    if torch.backends.mps.is_available():
        return torch.device("mps")  # Apple Silicon
    elif torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")

def load_model():
    """Load the pre-trained model."""
    global model, class_indices, idx_to_class, device, has_loaded_model
    
    try:
        # Set device
        device = get_device()
        
        # Check if model file exists
        model_path = 'best_model.pth'
        if not os.path.exists(model_path):
            print("Model file not found. Please run training first.")
            return False
        
        # Check for class indices
        class_indices_path = 'class_indices.npy'
        if not os.path.exists(class_indices_path):
            print("Class indices file not found. Please run training first.")
            return False
        
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        num_classes = checkpoint.get('num_classes', 38)  # Default to 38 if not specified
        
        # Create model
        model = models.efficientnet_b0(weights=None)
        num_ftrs = model.classifier[1].in_features
        model.classifier = nn.Sequential(
            nn.Dropout(p=0.5, inplace=True),
            nn.Linear(num_ftrs, num_classes)
        )
        
        # Load state dict
        model.load_state_dict(checkpoint.get('state_dict', checkpoint))
        model = model.to(device)
        model.eval()
        
        # Load class indices
        class_indices = np.load(class_indices_path, allow_pickle=True).item()
        idx_to_class = {v: k for k, v in class_indices.items()}
        
        has_loaded_model = True
        print(f"Model loaded successfully. Detected {num_classes} classes.")
        return True
            
    except Exception as e:
        print(f"Error loading model: {e}")
        return False

def predict_disease(image_path):
    """Analyze the uploaded image for plant diseases."""
    if not has_loaded_model:
        return {"error": "Model not loaded"}
    
    try:
        # Log that we're starting prediction
        print(f"Starting prediction for image: {image_path}")
        
        # Preprocess image
        try:
            transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
            ])
            
            print(f"Opening image file: {image_path}")
            image = Image.open(image_path).convert('RGB')
            print("Image opened successfully")
            
            image_tensor = transform(image).unsqueeze(0).to(device)
            print(f"Image tensor created with shape: {image_tensor.shape}")
        except Exception as img_e:
            print(f"Error in image preprocessing: {img_e}")
            return {"error": f"Image preprocessing failed: {str(img_e)}"}
        
        # Make prediction
        try:
            print("Running model inference...")
            with torch.no_grad():
                output = model(image_tensor)
                probabilities = torch.nn.functional.softmax(output, dim=1)[0]
                predicted_idx = torch.argmax(probabilities).item()
            print(f"Prediction complete, predicted index: {predicted_idx}")
        except Exception as model_e:
            print(f"Error in model inference: {model_e}")
            return {"error": f"Model inference failed: {str(model_e)}"}
        
        # Look up class name
        try:
            print(f"Getting class name for index {predicted_idx}")
            print(f"Available indices: {list(idx_to_class.keys())[:5]}...")
            class_name = idx_to_class[predicted_idx]
            confidence = probabilities[predicted_idx].item()
            print(f"Found class name: {class_name}, confidence: {confidence}")
        except Exception as idx_e:
            print(f"Error looking up class name: {idx_e}")
            return {"error": f"Class lookup failed: {str(idx_e)}"}
        
        # Parse plant and disease names
        parts = class_name.split('___')
        plant_name = parts[0]
        disease_name = parts[1] if len(parts) > 1 else "Unknown"
        
        # Get top 3 predictions for additional context
        top_probs, top_indices = torch.topk(probabilities, 3)
        top_predictions = [
            {"class": idx_to_class[i.item()].split('___')[-1] if '___' in idx_to_class[i.item()] else idx_to_class[i.item()],
             "probability": p.item() * 100} for i, p in zip(top_indices, top_probs)
        ]
        
        # Create response
        result = {
            "plant": plant_name,
            "disease": disease_name,
            "confidence": confidence * 100,
            "top_predictions": top_predictions,
            "is_healthy": "healthy" in disease_name.lower(),
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        }
        
        return result
        
    except Exception as e:
        print(f"Error analyzing image: {e}")
        return {"error": str(e)}

# Add context processor to inject variables into all templates
@app.context_processor
def inject_globals():
    """Inject global variables into all templates."""
    return {
        'now': datetime.now(),
        'model_loaded': has_loaded_model,
        'device': device
    }

# Load the model when the app starts
load_model()

@app.route('/')
def index():
    """Render the home page."""
    return render_template('index.html')

@app.route('/detect')
def detect():
    """Render the disease detection page."""
    return render_template('detect.html', model_loaded=has_loaded_model)

@app.route('/diseases')
def diseases():
    """Render the plant diseases information page."""
    return render_template('diseases.html')

@app.route('/about')
def about():
    """Render the about page."""
    return render_template('about.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    """Handle file upload and analysis."""
    # Check if model is loaded
    if not has_loaded_model:
        return jsonify({"error": "Model not loaded. Please check server logs."}), 500
    
    # Check if a file was uploaded
    if 'file' not in request.files:
        return jsonify({"error": "No file part"}), 400
    
    file = request.files['file']
    
    # Check if file is empty
    if file.filename == '':
        return jsonify({"error": "No selected file"}), 400
    
    # Check if file is allowed
    if file and allowed_file(file.filename):
        # Create a unique filename
        filename = str(uuid.uuid4()) + '_' + secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Analyze the image
        result = predict_disease(filepath)
        
        # Add the image URL to the result
        if "error" not in result:
            result["image_url"] = url_for('uploaded_file', filename=filename)
        
        return jsonify(result)
    
    return jsonify({"error": "File type not allowed"}), 400

@app.route('/uploads/<filename>')
def uploaded_file(filename):
    """Serve uploaded files."""
    return send_from_directory(app.config['UPLOAD_FOLDER'], filename)

@app.route('/model_status')
def model_status():
    """Return the status of the model."""
    return jsonify({
        "loaded": has_loaded_model,
        "device": str(device) if device else "Not set"
    })

@app.route('/train')
def train():
    """Redirect to training instructions."""
    return render_template('train.html')

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5001))
    app.run(host='0.0.0.0', port=port, debug=True)