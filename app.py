from flask import Flask, render_template, request, redirect, url_for
from model import lr_model, rf_model, predict_popularity

app = Flask(__name__)

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/predict', methods=['GET'])
def index():
    return render_template('index.html')

@app.route('/result', methods=['POST'])
def result():
    popularity_score = float(request.form['popularity_score'])
    year = int(request.form['year'])
    month = int(request.form['month'])
    model_type = request.form['model_type']
    model = lr_model if model_type == 'linear' else rf_model
    prediction = predict_popularity(model, popularity_score, year, month)
    return render_template('result.html', prediction=prediction)

if __name__ == '__main__':
    app.run(debug=True)
