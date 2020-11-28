import os
import sys

# Flask
from flask import Flask, redirect, url_for, request, render_template, Response, jsonify, redirect


# TensorFlow and tf.keras
import tensorflow as tf
from tensorflow import keras

from tensorflow.keras.applications.imagenet_utils import preprocess_input, decode_predictions
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing import image
import cv2
# Some utilites
import numpy as np
from util import base64_to_pil
from training_model import prepare
from tensorflow.keras.preprocessing.image import img_to_array


# You can use pretrained model from Keras
# Check https://keras.io/applications/
# from keras.applications.mobilenet_v2 import MobileNetV2
# model = MobileNetV2(weights='imagenet')

# print('Model loaded. Check http://127.0.0.1:5000/')


# Model saved with Keras model.save()
# Load your own trained model

def custom_predict(img):
    model = tf.keras.models.load_model("64x3-CNN.model")
    img = prepare(img)
    preds = model.predict(img)
    return preds


# Declare a flask app
app = Flask(__name__)


@app.route('/')
def index():
    # initModel()
    # render out pre-built HTML file right on the index page
    return render_template("index.html")


# Process images
@app.route("/predict", methods=["POST"])
def processreq():
    if request.method == 'POST':
        data = base64_to_pil(request.json)
        data.save("image.png")
        preds = custom_predict("image.png")
        resp = "{:.2f}".format(round(np.amax(preds)))
        resp = int(float(resp))
        if resp == 1:
            resp = "It seems to be a Dog"
        if resp == 0:
            resp = "It seems to be a Cat"
        return jsonify(result=resp)


if __name__ == "__main__":
    app.run(debug=True)
