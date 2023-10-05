from flask import Flask, render_template, request, jsonify
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import tensorflow as tf
import pickle
import json
import requests
import os

SERVER_URL = 'http://localhost:5000'
SAVE_DIR = 'global_model/'

    
def download_global_model():
    model_path = os.path.join(SAVE_DIR, 'model.h5')

    if os.path.exists(model_path):
        print('Global model already exists. Skipping download.')
        return
    
    response = requests.get(SERVER_URL + "/model")

    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    model_path = os.path.join(SAVE_DIR, 'model.h5')
    with open(model_path, 'wb') as model_file:
        model_file.write(response.content)

    print('Global model downloaded successfully.')

download_global_model()
tokenizer = pickle.load(open('tokenizer.pickle', 'rb'))
model = tf.keras.models.load_model('global_model/model.h5')

for layer in model.layers[:-1]:
    layer.trainable = False

app = Flask(__name__)

user_choices = []

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
    
def upload_updated_model():
    global SERVER_URL

    # Check if the model file exists
    if os.path.exists("global_model/model.h5"):
        # Upload the updated model to the server
        with open("global_model/model.h5", 'rb') as model_file:
            files = {'file': model_file}
            response = requests.post(f'{SERVER_URL}/upload_model', files=files)

        if response.status_code == 200:
            print('Updated model uploaded successfully to the server.')
        else:
            print('Failed to upload the updated model to the server.')
    else:
        print('Updated model file not found. Please ensure the file path is correct.')
    
def fine_tune_model(fine_tuning_data):
    try:
        model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(learning_rate=0.01), metrics=['accuracy'])

        input_sequences = []

        for line in fine_tuning_data:
            token_list = tokenizer.texts_to_sequences([line])[0];

            for i in range(1, len(token_list)):
                n_gram_sequence = token_list[:i+1]
                input_sequences.append(n_gram_sequence)

        if len(input_sequences) == 0:
            return

        total_words = len(tokenizer.word_index) + 1
        max_sequence_len = 6
        input_sequences = np.array(pad_sequences(input_sequences, maxlen=max_sequence_len, padding='pre'))

        xs, labels = input_sequences[:,:-1],input_sequences[:,-1]
        ys = tf.keras.utils.to_categorical(labels, num_classes=total_words)

        model.fit(xs, ys, epochs=1, verbose=1)
        print(xs.shape, labels.shape, ys.shape, "------ Model recompiled")
        model.save("global_model/model.h5")
        upload_updated_model()

    except Exception as ex:
        print(ex)


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        user_input = request.form['user_input']
        predicted_word = predict_next_word(user_input)
        return jsonify({'predicted_word': predicted_word})
    except Exception as ex:
        print(ex)



@app.route('/store_user_choice', methods=['POST'])
def store_user_choice():
    global user_choices
    try:
        data = json.loads(request.data)
        user_input = data['user_input']

        fine_tune_model([user_input])

        return jsonify({'message': 'User choice stored successfully'})
    except Exception as ex:
        print(ex)
        return jsonify({'message': 'An error occured'})


if __name__ == '__main__':
    try:
        app.run(port=5001)
    except Exception as ex:
        print(ex)
