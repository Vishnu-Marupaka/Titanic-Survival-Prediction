from flask import Flask, request, render_template
import joblib
import pandas as pd

app = Flask(__name__)
model = joblib.load('model.joblib')

@app.route('/', methods=['GET', 'POST'])
def home():
    prediction_text = ""
    
    if request.method == 'POST':
        try:
            age = float(request.form['Age'])
            sex = int(request.form['Sex'])
            pclass = int(request.form['Pclass'])
            fare = float(request.form['Fare'])

            data = pd.DataFrame({
                'Age': [age],
                'Sex': [sex],
                'Pclass': [pclass],
                'Fare': [fare]
            })

            prediction = model.predict(data)[0]
            
            if prediction == 1:
                result = "Survived"
            else:
                result = "Did Not Survive"
            
            prediction_text = f"Prediction: {result}"
            
        except Exception as e:
            prediction_text = f"Error: {str(e)}"

    return render_template('index.html', result=prediction_text)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=7860)