from flask import Flask, request, jsonify, send_from_directory
import pandas as pd
import joblib

app = Flask(__name__)

model = joblib.load("student_marks_model.pkl")


@app.route("/")
def home():
    return send_from_directory(".", "index.html")


@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        hours = float(data["hours"])
        previous = float(data["previous"])
        sleep = float(data["sleep"])

        input_data = pd.DataFrame(
            [[hours, previous, sleep]],
            columns=[
                "Hours_Studied",
                "Previous_Marks",
                "Sleep_Hours"
            ]
        )

        prediction = model.predict(input_data)[0]

        return jsonify({
            "prediction": round(float(prediction), 2)
        })

    except Exception as e:
        return jsonify({"error": str(e)}), 400


if __name__ == "__main__":
    app.run(debug=True)