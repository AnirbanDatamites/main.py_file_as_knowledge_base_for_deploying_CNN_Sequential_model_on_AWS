import os
import uuid
import pickle
from flask import Flask, request, jsonify, render_template
import numpy as np
from tensorflow.keras.preprocessing.image import load_img, img_to_array

app = Flask(__name__)

# Load your trained model from the pickle file
MODEL_PATH = 'modelcnn.pkl'
with open(MODEL_PATH, 'rb') as model_file:
    model = pickle.load(model_file)

# Define the allowed extensions
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part in the request'}), 400

    file = request.files['file']

    if file.filename == '':
        return jsonify({'error': 'No file selected for uploading'}), 400

    if file and allowed_file(file.filename):
        # Create the static directory if it doesn't exist
        if not os.path.exists('static'):
            os.makedirs('static')

        # Generate a unique filename
        filename = f"{uuid.uuid4().hex}.jpg"
        file_path = os.path.join('static', filename)

        # Save the file
        file.save(file_path)

        # Preprocess the image
        image = load_img(file_path, target_size=(28, 28), color_mode="grayscale")
        image = img_to_array(image)
        image = image.reshape(1, 28, 28, 1)
        image = image.astype('float32') / 255.0

        # Make a prediction
        predictions = model.predict(image)
        predicted_class = np.argmax(predictions, axis=1)[0]

        # Define class labels (adjust according to your model)
        class_labels = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat', 'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']
        result = class_labels[predicted_class]

        return jsonify({'prediction': result, 'filename': filename})

    else:
        return jsonify({'error': 'Allowed file types are png, jpg, jpeg'}), 400

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=8080)

#You need to modify the MODEL_PATH according to your path on your local machine and also change the port in the end instead of port=8080 as your requirements