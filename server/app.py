# create a flask application with a simple get endpoint that returns the model in folder model, this is a LSTM model for next word prediction
# the model is trained on a corpus of 1000 books from project Gutenberg

import os
from flask import Flask, send_from_directory, request, jsonify

app = Flask(__name__, static_url_path='')

MODEL_FOLDER = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'model')

# make a get endpoint at  /, that just returns a string to know that the server is running
@app.route('/', methods=['GET'])
def index():
    return 'Server is running'

@app.route('/model', methods=['GET'])
def get_model():
    try:
        return send_from_directory(MODEL_FOLDER, 'model.h5')
    except Exception as e:
        print(e)
        return jsonify({'error': 'model not found'}), 404
    
if __name__ == '__main__':
    app.run( port=5000, debug=True)