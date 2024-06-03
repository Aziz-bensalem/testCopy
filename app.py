from flask import Flask, render_template, request, jsonify
import numpy as np
from PIL import Image
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.models import load_model

app = Flask(__name__)

model = load_model("./mymodel.h5")

def predict_image(image_file):
    classes = ['Actinic keratoses', 'Basal cell carcinoma',
               'Benign keratosis-like lesions', 'Dermatofibroma', 'Melanoma',
               'Melanocytic nevi', 'Vascular lesions']
    le = LabelEncoder()
    le.fit(classes)

    SIZE = 64
    img = np.asarray(Image.open(image_file).resize((SIZE, SIZE)))
    img = img / 255.
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)[0]  # Take the first element since we only have one prediction
    confidence_scores = {cls: round(float(confidence) * 100, 2) for cls, confidence in zip(classes, pred)}
    return confidence_scores

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def handle_prediction():
    if request.method == 'POST':
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded'}), 400

        image = request.files['image']
        if image.filename == '':
            return jsonify({'error': 'No selected file'}), 400

        # Process and predict
        confidence_scores = predict_image(image)
        return jsonify({'confidence_scores': confidence_scores})

    return jsonify({'error': 'Method not allowed'}), 405

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
