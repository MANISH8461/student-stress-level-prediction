# Student Stress Level Prediction 🎓

A Flask web app that predicts whether a student is experiencing **High Stress**
or **Normal Stress** based on lifestyle and academic factors, using a
**Logistic Regression** model.

🔗 **Live App:** [Try it live here](https://student-stress-flask.onrender.com)
*(redeploy required after the fixes below — see Deployment note)*

> **Note:** This app runs on Render's free tier, which "sleeps" after ~15
> minutes of inactivity. The first request after sleeping can take 30-50
> seconds to wake up — this is normal, not a bug.

---

## 📋 About the Project

This app takes in details about a student's daily routine — sleep, study
hours, social media usage, attendance, exam pressure, and family support — and
predicts their stress level using a Logistic Regression model trained on a
public student lifestyle dataset (25,500 raw rows, cleaned to 16,721).

**Read [`case-study.md`](./case-study.md) for the full development history:**
eight real bugs were found and fixed during this project — including a silent
encoding mismatch between training and the deployed app, leakage inside
cross-validation, and a variable-shadowing bug that caused two inconsistent
pipelines to coexist in the same notebook. Nothing below is asserted without
having been checked against the actual code, data, or a baseline comparison.

## 🧠 Model Details

- **Algorithm:** Logistic Regression (`C=0.1`, `class_weight='balanced'`)
- **Preprocessing:** `StandardScaler` (feature scaling) + `LabelEncoder`
  (Student_Type), both fit on the training split only
- **Target:** Binary classification — `1` = High Stress, `0` = Normal Stress
- **Class balance:** ~70% Normal / ~30% High Stress after cleaning

### Performance (held-out 20% test split, 3,345 rows)

| Metric | Value |
|---|---|
| Accuracy | 0.7955 |
| F1 (High Stress) | 0.7067 |
| Recall (High Stress) | 0.8047 |
| Precision (High Stress) | 0.63 |
| Majority-class baseline accuracy | 0.7037 |

Logistic Regression was chosen after comparing 7 models (KNN, Naive Bayes,
Decision Tree, SVM, Random Forest, XGBoost) via `GridSearchCV(cv=5,
scoring='f1')` inside proper `Pipeline`s (scaler refit per fold, no leakage).
Four models — Random Forest, SVM, Logistic Regression, XGBoost — scored within
0.0023 F1 of each other on cross-validation, a gap smaller than run-to-run
noise. Logistic Regression was selected from that tie for its native
calibrated `predict_proba`, lower serving cost, and directly interpretable
coefficients — not because it had the single highest score. See
`case-study.md` for the full comparison table.

### Known limitations

- **Precision is 0.63** on the High Stress class — about 37% of positive
  flags are false positives. Fine for an informational signal; not suitable
  as-is for triggering automatic interventions without disclosing this rate.
- **Exam_Pressure (self-reported 1–10) dominates the model** (correlation
  0.52 with the target, largest coefficient by a wide margin). This model is
  better described as predicting stress from a mix of self-reported pressure
  and lifestyle habits, not purely from behavioral lifestyle factors alone.
- **Student_Type has near-zero independent effect** (coefficient -0.0037)
  once other features are included, despite a real raw difference in stress
  rates by type (school 20.5%, college 31.3%, working student 38.2%) — it
  appears to be a proxy for other included features rather than an
  independent predictor.

## 📂 Project Files

| File | Description |
|---|---|
| `app.py` | Flask application (routes + prediction logic) |
| `templates/index.html` | HTML form (UI), rendered by Flask |
| `logisticReg_Student_Stress_Level.pkl` | Trained Logistic Regression model |
| `scaler_studentLogistic_.pkl` | StandardScaler fitted on training data only |
| `columns_studentStress.pkl` | Expected feature column order for the model |
| `studentType_encoder.pkl` | LabelEncoder for Student_Type, fitted on training data only |
| `case-study.md` | Full record of bugs found and fixed during development |
| `requirements.txt` | Python dependencies |
| `Procfile` | Tells Render how to run the app with Gunicorn |

## ⚙️ Input Features

| Feature | Description | Valid range |
|---|---|---|
| Student Type | School / College / Working Student | fixed options |
| Sleep Hours | Average daily sleep hours | 2–10 |
| Study Hours | Average daily study hours | 0–22 |
| Social Media Hours | Average daily social media usage | 4–10 |
| Attendance | Attendance percentage | 0–100 |
| Exam Pressure | Self-rated exam pressure | 1–10 |
| Family Support | Self-rated family support | 1–10 |
| Month | Month of the academic year | 1–12 |

## 🚀 Run Locally

1. Clone this repository:
   ```bash
   git clone https://github.com/MANISH8461/student-stress-level-prediction.git
   cd student-stress-level-prediction
   ```

2. Create a virtual environment and activate it:
   ```bash
   python -m venv venv
   venv\Scripts\activate      # Windows
   source venv/bin/activate   # macOS/Linux
   ```

3. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

4. Run the app:
   ```bash
   python app.py
   ```

5. Open **http://127.0.0.1:5000** in your browser, fill in the form, and get a
   prediction.

## 🌐 Deploying to Render (free live link)

1. Push this project to GitHub (including all `.pkl` files — model, scaler,
   columns, **and encoder** — plus `Procfile` and `requirements.txt`).
2. Go to [render.com](https://render.com) and sign up (free tier is fine).
3. Click **New → Web Service**, connect your GitHub account, and select this
   repo.
4. Fill in the settings:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. Click **Create Web Service**. Render installs dependencies and gives you a
   live URL like `https://your-app-name.onrender.com`.

> **⚠️ Redeploy note:** if you had a previous live deployment, it was built
> from a model with a `Student_Type` encoding mismatch (see `case-study.md`,
> Bug 2) — School and College were silently swapped for every prediction. The
> currently deployed model needs to be replaced with the artifacts produced
> after the Bug 2–8 fixes before the live link reflects correct behavior.

## 🖥️ Code Overview

```python
# Load the trained model, scaler, column order, and encoder once, at startup
model = joblib.load("logisticReg_Student_Stress_Level.pkl")
scaler = joblib.load("scaler_studentLogistic_.pkl")
expected_columns = joblib.load("columns_studentStress.pkl")
student_type_encoder = joblib.load("studentType_encoder.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        raw_input = {
            "Sleep_Hours": sleep_hours,
            "Study_Hours": study_hours,
            "Social_Media_Hours": social_media_hours,
            "Attendance": attendance,
            "Exam_Pressure": exam_pressure,
            "Family_Support": family_support,
            "Month": month,
            "Student_Type": student_type_encoder.transform(
                [student_type.lower().replace(" ", "_")]
            )[0],
        }
        input_df = pd.DataFrame([raw_input])
        input_df = input_df[expected_columns]  # match training column order

        scaled_input = scaler.transform(input_df)
        prediction = model.predict(scaled_input)[0]
        result = "High Stress Level" if prediction == 1 else "Normal Stress Level"

    return render_template("index.html", prediction=result)
```

## 🛠️ Tech Stack

- Python
- Flask
- scikit-learn
- pandas
- joblib
- Gunicorn (production server)
- Bootstrap 5 (frontend styling)

## 👤 Author

**Manish**

---

⭐ If you found this project helpful, consider giving it a star on GitHub!