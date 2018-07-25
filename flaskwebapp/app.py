import os
from flask import Flask
from flask import jsonify
from serve import *

app = Flask(__name__)

@app.route('/api/test')
def test():
    return jsonify(result = "ok")

@app.route('/api/test2')
def test2():
    raw_X, processed_X = get_4_samples()
    predictions = serve_model(processed_X)
    return jsonify(result = predictions)

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)