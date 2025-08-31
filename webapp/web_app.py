from flask import Flask, request, render_template, jsonify
from fastai.vision.all import *
import os
from pathlib import Path

app = Flask(__name__)
app.config['UPLOAD_FOLDER'] = 'uploads'
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max file size

# Create uploads directory
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Load the trained model
learn = load_learner('export.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file uploaded'})
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'})
    
    if file and file.filename.lower().endswith(('.png', '.jpg', '.jpeg')):
        # Save uploaded file
        filename = file.filename
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        # Make prediction
        try:
            pred_class, pred_idx, probs = learn.predict(filepath)
            confidence = float(probs.max())
            
            # Get all probabilities
            all_probs = {class_name: float(prob) for class_name, prob in zip(learn.dls.vocab, probs)}
            
            return jsonify({
                'prediction': str(pred_class),
                'confidence': confidence,
                'all_probabilities': all_probs
            })
        except Exception as e:
            return jsonify({'error': f'Prediction failed: {str(e)}'})
    
    return jsonify({'error': 'Invalid file type. Please upload PNG, JPG, or JPEG'})

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)