# create a flask application with a simple get endpoint that returns the model in folder model, this is a LSTM model for next word prediction
# the model is trained on a corpus of 1000 books from project Gutenberg

import os
from flask import Flask, send_from_directory, request, jsonify, render_template
import pickle
import tensorflow as tf
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

app = Flask(__name__, static_url_path='')

MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
model = tf.keras.models.load_model(MODEL_FOLDER + '/model.h5')
# make a get endpoint at  /, that just returns a string to know that the server is running
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/model', methods=['GET'])
def get_model():
    try:
        return send_from_directory(MODEL_FOLDER, 'model.h5')
    except Exception as e:
        print(e)
        return jsonify({'error': 'model not found'}), 404
    
@app.route('/upload_model', methods=['POST'])
def upload_model():
    global MODEL_FOLDER

    # Check if the 'file' key is in the request files
    if 'file' not in request.files:
        return jsonify({'message': 'No file part'}), 400

    file = request.files['file']
    if file.filename == '':
        return jsonify({'message': 'No selected file'}), 400

    model_path = os.path.join(MODEL_FOLDER, 'model.h5')
    file.save(model_path)

    return jsonify({'message': 'Updated model uploaded successfully'}), 200

def predict_next_word(seed_text):
    try:
        predict_word = ""

        token_list = tokenizer.texts_to_sequences([seed_text])[0]
        token_list = pad_sequences([token_list], maxlen=6-1, padding='pre')
        predicted = model.predict(token_list)
        predicted = np.argmax(predicted, axis=1)

        for word, index in tokenizer.word_index.items():
            if index == predicted:
                predict_word += word
        return predict_word
    
    except Exception as ex:
        print(ex)
        return ""

@app.route('/predict', methods=['POST'])
def predict():
    try:
        print(request.form)
        user_input = request.form['user_input']
        predicted_word = predict_next_word(user_input)
        return jsonify({'predicted_word': predicted_word})
    except Exception as ex:
        print(ex)
    
if __name__ == '__main__':
    app.run( port=5000)