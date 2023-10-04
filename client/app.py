import requests
import os
import tensorflow as tf
import pickle
import numpy as np
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences

# URL of the server where the global model is hosted
SERVER_URL = 'http://localhost:5000/model'  # Replace with the actual server IP and port

# Directory to save the downloaded model
SAVE_DIR = 'global_model/'

# Load the tokenizer
with open('tokenizer.pickle', 'rb') as handle:
    tokenizer = pickle.load(handle)
    print(tokenizer)

def download_global_model():
    response = requests.get(SERVER_URL)

    # Ensure the save directory exists
    if not os.path.exists(SAVE_DIR):
        os.makedirs(SAVE_DIR)

    # Save the received model to the client's directory
    model_path = os.path.join(SAVE_DIR, 'model.h5')
    with open(model_path, 'wb') as model_file:
        model_file.write(response.content)

    print('Global model downloaded successfully.')


def load_global_model():
    # Load the model
    model = tf.keras.models.load_model(os.path.join(SAVE_DIR, 'model.h5'))
    return model

def predict_next_word(seed_text, mode, tokenizer):
    token_list = tokenizer.texts_to_sequences([seed_text])[0]
    token_list = pad_sequences([token_list], maxlen=40-1, padding='pre')
    predicted=model.predict(token_list)
    predicted=np.argmax(predicted,axis=1)
    output_word = ""

    for word, index in tokenizer.word_index.items():
        if index == predicted:
            output_word = word
            break
    return output_word

if __name__ == '__main__':
    download_global_model()
    model = load_global_model()

    seed_texts = ["This is", "Can you", "How", "What about", "How are", "I do", "Bananas", "Apples", "I love"]
    for seed_text in seed_texts:
        predicted_word = predict_next_word(seed_text, model, tokenizer)
        print(f'Next predicted word after "{seed_text}": {predicted_word}')

