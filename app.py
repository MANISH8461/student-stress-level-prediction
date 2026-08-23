"""
app.py
------
Flask web app for the Student Stress Level classifier.
Loads the trained model, scaler, expected column order, and Student_Type
encoder, and serves a form where a user enters lifestyle details and gets a
High Stress / Normal Stress prediction.

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
# Load the trained model, scaler, expected columns, and encoder once, at startup.
# All four artifacts must come from the same training run — see case-study.md
# Bug 5 for why mixing artifacts from different runs caused problems before.
# ---------------------------------------------------------
model = joblib.load("logisticReg_Student_Stress_Level.pkl")
scaler = joblib.load("scaler_studentLogistic_.pkl")
expected_columns = joblib.load("columns_studentStress.pkl")
student_type_encoder = joblib.load("studentType_encoder.pkl")

STUDENT_TYPES = ["School", "College", "Working Student"]


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

            # Encoder was fit on lowercase, underscore-separated strings
            # (e.g. "working_student"). Normalize form values to match.
            encoded_student_type = student_type_encoder.transform(
                [student_type.lower().replace(" ", "_")]
            )[0]

            raw_input = {
                "Sleep_Hours": sleep_hours,
                "Study_Hours": study_hours,
                "Social_Media_Hours": social_media_hours,
                "Attendance": attendance,
                "Exam_Pressure": exam_pressure,
                "Family_Support": family_support,
                "Month": month,
                "Student_Type": encoded_student_type,
            }

            input_df = pd.DataFrame([raw_input])
            input_df = input_df[expected_columns]  # match training column order

            scaled_input = scaler.transform(input_df)
            result = model.predict(scaled_input)[0]
            probabilities = model.predict_proba(scaled_input)[0]

            print("Input DataFrame:\n", input_df)
            print("Scaled Input:\n", scaled_input)
            print("Prediction:", result)
            print("Probabilities:", probabilities)

            prediction = "🚨 High Stress Level" if result == 1 else "✅ Normal Stress Level"
            confidence = round(max(probabilities) * 100, 1)  # confidence in the predicted class
        except Exception as e:
            error = f"Invalid input: {e}"

    return render_template(
        "index.html",
        student_types=STUDENT_TYPES,
        prediction=prediction,
        confidence=confidence,
        error=error,
    )


if __name__ == "__main__":
    # debug=True only for local testing — Render runs this via gunicorn (see Procfile)
    port = int(os.environ.get("PORT", 5000))
    app.run(debug=True, host="0.0.0.0", port=port)