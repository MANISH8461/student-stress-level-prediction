# Student Stress Level Prediction 🎓

A Flask web app that predicts whether a student is experiencing **High Stress** or **Normal Stress** based on lifestyle and academic factors, using a **Logistic Regression** model.

🔗 **Live App:** [Try it live here](https://student-stress-flask.onrender.com)

> **Note:** This app runs on Render's free tier, which "sleeps" after ~15
> minutes of inactivity. The first request after sleeping can take 30-50
> seconds to wake up — this is normal, not a bug.

---

## 📋 About the Project

This app takes in details about a student's daily routine — sleep, study hours, social media usage, attendance, exam pressure, and family support — and predicts their stress level using a machine learning model trained on student lifestyle data.

## 🧠 Model Details

- **Algorithm:** Logistic Regression
- **Preprocessing:** StandardScaler (feature scaling)
- **Target:** Binary classification — `1` = High Stress, `0` = Normal Stress

## 📂 Project Files

| File | Description |
|---|---|
| `app.py` | Flask application (routes + prediction logic) |
| `templates/index.html` | HTML form (UI), rendered by Flask |
| `logisticReg_Student_Stress_Level.pkl` | Trained Logistic Regression model |
| `scaler_studentLogistic_.pkl` | StandardScaler fitted on training data |
| `columns_studentStress.pkl` | Expected feature column order for the model |
| `requirements.txt` | Python dependencies |
| `Procfile` | Tells Render how to run the app with Gunicorn |

## ⚙️ Input Features

| Feature | Description |
|---|---|
| Student Type | School / College / Working Student |
| Sleep Hours | Average daily sleep hours |
| Study Hours | Average daily study hours |
| Social Media Hours | Average daily social media usage |
| Attendance | Attendance percentage |
| Exam Pressure | Self-rated exam pressure (1–10) |
| Family Support | Self-rated family support (1–10) |
| Month | Month of the academic year (1–12) |

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

5. Open **http://127.0.0.1:5000** in your browser, fill in the form, and get a prediction.

## 🌐 Deploying to Render (free live link)

1. Push this project to GitHub (including the `.pkl` model files, `Procfile`, and `requirements.txt`).
2. Go to [render.com](https://render.com) and sign up (free tier is fine).
3. Click **New → Web Service**, connect your GitHub account, and select this repo.
4. Fill in the settings:
   - **Runtime:** Python 3
   - **Build Command:** `pip install -r requirements.txt`
   - **Start Command:** `gunicorn app:app`
   - **Instance Type:** Free
5. Click **Create Web Service**. Render installs dependencies and gives you a live URL like `https://your-app-name.onrender.com`.

## 🖥️ Code Overview

```python
# Load the trained model, scaler, and expected column order once, at startup
model = joblib.load("logisticReg_Student_Stress_Level.pkl")
scaler = joblib.load("scaler_studentLogistic_.pkl")
expected_columns = joblib.load("columns_studentStress.pkl")

@app.route("/", methods=["GET", "POST"])
def home():
    if request.method == "POST":
        # Collect form input and build a single-row DataFrame
        raw_input = {...}
        input_df = pd.DataFrame([raw_input])

        # Reorder columns to match training order exactly
        input_df = input_df[expected_columns]

        # Scale input the same way training data was scaled
        scaled_input = scaler.transform(input_df)

        # Predict and display result
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