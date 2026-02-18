from flask import Flask, render_template, request
from recognise import predict_gesture
import os

app = Flask(__name__)

@app.route('/')
def home():
    return render_template("index.html")

@app.route('/predict', methods=['POST'])
def predict():
    file = request.files['file']
    filepath = os.path.join("uploads", file.filename)
    file.save(filepath)

    result = predict_gesture(filepath)
    return render_template("index.html", prediction=result)

if __name__ == "__main__":
    app.run(debug=True)
