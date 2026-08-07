"""
app.py
------
Flask web app for the Student Stress Level classifier.
Loads the trained model, scaler, and expected column order,
and serves a form where a user enters lifestyle details and
gets a High Stress / Normal Stress prediction.

Run with:
    python app.py
Then open http://127.0.0.1:5000 in your browser.
"""

from flask import Flask, render_template, request
import pandas as pd
import joblib
import os

app = Flask(__name__)

# ---------------------------------------------------------
# Load the trained model, scaler, and expected columns once, at startup
# ---------------------------------------------------------
model = joblib.load("logisticReg_Student_Stress_Level.pkl")
scaler = joblib.load("scaler_studentLogistic_.pkl")
expected_columns = joblib.load("columns_studentStress.pkl")

STUDENT_TYPES = ["School", "College", "Working Student"]
STUDENT_TYPE_MAP = {"School": 0, "College": 1, "Working Student": 2}


@app.route("/", methods=["GET", "POST"])
def home():
    prediction = None
    error = None

    if request.method == "POST":
        try:
            student_type = request.form["studentType"]
            sleep_hours = float(request.form["sleepHours"])
            study_hours = float(request.form["studyHours"])
            social_media_hours = float(request.form["socialMediaHours"])
            attendance = float(request.form["attendance"])
            exam_pressure = float(request.form["examPressure"])
            family_support = float(request.form["familySupport"])
            month = int(request.form["month"])

            raw_input = {
                "Sleep_Hours": sleep_hours,
                "Study_Hours": study_hours,
                "Social_Media_Hours": social_media_hours,
                "Attendance": attendance,
                "Exam_Pressure": exam_pressure,
                "Family_Support": family_support,
                "Month": month,
                "Student_Type": STUDENT_TYPE_MAP[student_type],
            }

            input_df = pd.DataFrame([raw_input])

            # Fill any expected column not present in raw_input with 0
            for col in expected_columns:
                input_df = input_df[expected_columns]
                if col not in input_df.columns:
                    input_df[col] = 0

            # Reorder to match training column order exactly
            scaled_input = scaler.transform(input_df)
            result = model.predict(scaled_input)[0]
            probabilities = model.predict_proba(scaled_input)[0]

            print("Input DataFrame:\n", input_df)
            print("Scaled Input:\n", scaled_input)
            print("Prediction:", result)
            print("Probabilities:", probabilities)

            prediction = "🚨 High Stress Level" if result == 1 else "✅ Normal Stress Level"

        except Exception as e:
            error = f"Invalid input: {e}"

    return render_template(
        "index.html",
        student_types=STUDENT_TYPES,
        prediction=prediction,
        error=error,
    )


if __name__ == "__main__":
    # debug=True only for local testing — Render runs this via gunicorn (see Procfile)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)