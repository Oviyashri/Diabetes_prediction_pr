# Importing essential libraries
from flask import Flask, render_template, request
import joblib
import numpy as np

# Load the Random Forest CLassifier model
model = joblib.load(open("models/model.pkl", 'rb'))

app = Flask(__name__)

@app.route('/',methods=['GET'])
def home():
	return render_template('index.html')

@app.route('/prediction', methods=['POST'])
def prediction():
    if request.method == 'POST':
        preg = int(request.form['pregnancies'])
        glucose = int(request.form['glucose'])
        bp = int(request.form['bloodpressure'])
        st = int(request.form['skinthickness'])
        insulin = int(request.form['insulin'])
        bmi = float(request.form['bmi'])
        dpf = float(request.form['dpf'])
        age = int(request.form['age'])
        
        arr = np.array([[preg, glucose, bp, st, insulin, bmi, dpf, age]])
        prediction = model.predict(arr)
        
    return render_template('result.html', data=prediction)

if __name__ == "__main__":
	app.run()