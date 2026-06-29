from flask import Flask, render_template, request
import tensorflow as tf
import numpy as np
import cv2
import os

app = Flask(__name__)

# Loading the trained model
model = tf.keras.models.load_model(
    "image_classifier.h5"
)

classes = [
    'airplane', 'automobile', 'bird', 'cat',
    'deer', 'dog', 'frog', 'horse',
    'ship', 'truck'
]

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():

    file = request.files['image']

    filepath = os.path.join(
        "static",
        file.filename
    )

    file.save(filepath)

    img = cv2.imread(filepath)
    img = cv2.resize(img, (32, 32))
    img = img / 255.0

    img = np.expand_dims(img, axis=0)

    prediction = model.predict(img)

    class_index = np.argmax(prediction)
    result = classes[class_index]

    confidence = round(
        np.max(prediction) * 100,
        2
    )

    return render_template(
        'index.html',
        prediction=result,
        confidence=confidence,
        image_path=filepath
    )

if __name__ == "__main__":
    app.run(debug=True)
