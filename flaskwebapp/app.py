import os
import sys
from flask import Flask, render_template, flash, abort,redirect, request,g,url_for, session, make_response, jsonify, send_file
from serve import *
import random

app = Flask(__name__)
app.secret_key = "super secret key"

@app.errorhandler(404)
def not_found(error):
    return make_response(jsonify({'error': 'Not found'}), 404)

@app.route('/api/test')
def test():
    return jsonify(result = "ok")

@app.route('/api/test2')
def test2():
    raw_X, processed_X = get_4_samples()
    predictions = serve_model_mnist(processed_X)
    return jsonify(result = str(predictions))

@app.route('/')
@app.route('/index')
def begin():
    return render_template('base.html')

@app.route('/mnist')
def mnist():
    random_index = random.choices(range(10000),k=4)
    f_path = './static/images/' + '-'.join([str(x) for x in random_index]) + '.png'
    plot, prediction = serve_model_mnist(random_index)
    plot.savefig(f_path)
    flash('The predicted digits are ' + prediction)
    return render_template('mnist.html', url = f_path)

@app.route('/sentiment')
def sentiment():
    return render_template('sentiment.html')

@app.route('/submit_number', methods = ['POST'])
def submit_number():
    return redirect('/mnist')
    
@app.route('/submit_text', methods = ['POST'])
def submit_text():
    input_text = request.form['input_text']
    result = serve_model_sentiment(input_text)
    flash('The predicted star is ' + result)
    return redirect('/sentiment')

if __name__ == '__main__':
    app.run(host='0.0.0.0',port=80)