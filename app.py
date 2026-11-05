from flask import Flask, request
import numpy as np
import joblib

app = Flask(__name__)

# Load trained model
model = joblib.load('student_marks_model.pkl')

HTML_PAGE = """
<!DOCTYPE html>
<html>
<head>
    <title>Student Marks Prediction</title>
    <meta name="viewport" content="width=device-width, initial-scale=1.0">

    <style>
        body {
            background-color: #050c09;
            color: white;
            font-family: Arial;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
            margin: 0;
        }

        .card {
            background: #0b1a12;
            border: 1px solid #00ff88;
            border-radius: 20px;
            padding: 25px;
            width: 90%;
            max-width: 400px;
            text-align: center;
            box-shadow: 0 0 25px rgba(0,255,136,0.3);
        }

        h1 {
            font-size: 23px;
        }

        label {
            display: block;
            text-align: left;
            color: #00ff88;
            margin-top: 15px;
        }

        input {
            width: 100%;
            padding: 12px;
            box-sizing: border-box;
            background: #050c09;
            color: white;
            border: 1px solid #00ff88;
            border-radius: 8px;
            margin-top: 5px;
        }

        button {
            width: 100%;
            padding: 13px;
            margin-top: 20px;
            background: #00ff88;
            color: black;
            border: none;
            border-radius: 25px;
            font-weight: bold;
            cursor: pointer;
        }

        .result-box {
            margin-top: 20px;
            color: #00ff88;
            padding: 10px;
            border: 1px solid #00ff88;
            border-radius: 8px;
        }
    </style>
</head>

<body>
    <div class="card">
        <h1>🎓 Student Marks Prediction</h1>
        <p>AI Based Performance Predictor</p>

        <form action="/predict" method="POST">
            <label>Hours Studied</label>
            <input type="number" step="any" name="hours" required>

            <label>Previous Marks</label>
            <input type="number" step="any" name="previous_marks" required>

            <label>Sleep Hours</label>
            <input type="number" step="any" name="sleep" required>

            <button type="submit">🚀 Predict Marks</button>
        </form>

        RESULT_PLACEHOLDER
    </div>
</body>
</html>
"""


@app.route('/')
def home():
    return HTML_PAGE.replace("RESULT_PLACEHOLDER", "")


@app.route('/predict', methods=['POST'])
def predict():
    try:
        hours = float(request.form['hours'])
        previous_marks = float(request.form['previous_marks'])
        sleep = float(request.form['sleep'])

        features = np.array([[hours, previous_marks, sleep]])
        prediction = model.predict(features)

        output = round(float(prediction[0]), 2)

        result = f"""
        <div class="result-box">
            <h3>Predicted Marks: {output}</h3>
        </div>
        """

        return HTML_PAGE.replace("RESULT_PLACEHOLDER", result)

    except Exception:
        result = """
        <div class="result-box">
            <h3>Prediction Error! Please check your inputs.</h3>
        </div>
        """
        return HTML_PAGE.replace("RESULT_PLACEHOLDER", result)


if __name__ == '__main__':
    app.run(debug=False)