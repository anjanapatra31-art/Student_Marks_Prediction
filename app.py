from flask import Flask, render_template_string, request
import numpy as np
import pickle

app = Flask(__name__)

# Model load karein
model = pickle.load(open('model.pkl', 'rb'))

HTML_PAGE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Student Marks Predictor</title>
    <style>
        body {
            background-color: #050c09;
            color: #ffffff;
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            margin: 0;
            padding: 20px;
            display: flex;
            justify-content: center;
            align-items: center;
            min-height: 100vh;
        }
        .card {
            background: #0b1a12;
            border: 1px solid #00ff88;
            box-shadow: 0 0 25px rgba(0, 255, 136, 0.3);
            border-radius: 20px;
            padding: 25px;
            width: 100%;
            max-width: 400px;
            text-align: center;
        }
        h1 { font-size: 22px; color: #ffffff; margin-bottom: 5px; }
        .subtitle { font-size: 12px; color: #8fa396; margin-bottom: 20px; }
        .input-group { text-align: left; margin-bottom: 12px; }
        .input-group label { display: block; font-size: 13px; color: #00ff88; margin-bottom: 4px; }
        .input-group input {
            width: 100%;
            padding: 10px;
            background: #050c09;
            border: 1px solid #1e3d2f;
            border-radius: 8px;
            color: #ffffff;
            font-size: 15px;
            box-sizing: border-box;
            outline: none;
        }
        .predict-btn {
            background: linear-gradient(135deg, #00ff88, #00b359);
            color: #000000;
            font-weight: bold;
            font-size: 15px;
            border: none;
            padding: 12px;
            width: 100%;
            border-radius: 25px;
            cursor: pointer;
            margin-top: 10px;
        }
        .result-box {
            margin-top: 15px;
            background: #050c09;
            border: 1px solid #00ff88;
            padding: 10px;
            border-radius: 8px;
        }
        .result-box h3 { margin: 0; color: #00ff88; font-size: 16px; }
    </style>
</head>
<body>
    <div class="card">
        <div style="font-size: 35px;">🎓</div>
        <h1>Student Marks Prediction</h1>
        <p class="subtitle">AI Based Performance Predictor</p>
        
        <form action="/predict" method="POST">
            <div class="input-group">
                <label>Hours Studied</label>
                <input type="number" step="any" name="hours" placeholder="e.g. 6" required>
            </div>
            
            <div class="input-group">
                <label>Previous Marks</label>
                <input type="number" step="any" name="previous_marks" placeholder="e.g. 80" required>
            </div>

            <div class="input-group">
                <label>Sleep Hours</label>
                <input type="number" step="any" name="sleep" placeholder="e.g. 8" required>
            </div>

            <button type="submit" class="predict-btn">🚀 Predict Marks</button>
        </form>

        {% if prediction_text %}
            <div class="result-box">
                <h3>{{ prediction_text }}</h3>
            </div>
        {% endif %}
    </div>
</body>
</html>
"""

@app.route('/')
def home():
    return render_template_string(HTML_PAGE, prediction_text="")

@app.route('/predict', methods=['POST'])
def predict():
    try:
        float_features = [float(x) for x in request.form.values()]
        final_features = [np.array(float_features)]
        prediction = model.predict(final_features)
        output = round(prediction[0], 2)
        return render_template_string(HTML_PAGE, prediction_text=f"Predicted Marks: {output}")
    except Exception as e:
        return render_template_string(HTML_PAGE, prediction_text="Error in input values.")

if __name__ == '__main__':
    app.run(debug=True)