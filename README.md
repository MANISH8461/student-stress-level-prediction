# Student Stress Level Prediction 🎓

A Streamlit web app that predicts whether a student is experiencing **High Stress** or **Normal Stress** based on lifestyle and academic factors, using a **Logistic Regression** model.

🔗 **Live App:** [student-stress-level-prediction](https://student-stress-level-prediction-e3z4gtasvaqp5nfjwm9mpm.streamlit.app)

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
| `app.py` | Streamlit application (UI + prediction logic) |
| `logisticReg_Student_Stress_Level.pkl` | Trained Logistic Regression model |
| `scaler_studentLogistic_.pkl` | StandardScaler fitted on training data |
| `columns_studentStress.pkl` | Expected feature column order for the model |
| `requirements.txt` | Python dependencies |

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

```bash
git clone https://github.com/MANISH8461/student-stress-level-prediction.git
cd student-stress-level-prediction
pip install -r requirements.txt
streamlit run app.py
```

## 🖥️ Code Overview

```python
# Load the trained model, scaler, and expected column order
model = joblib.load("logisticReg_Student_Stress_Level.pkl")
scaler = joblib.load("scaler_studentLogistic_.pkl")
expected_columns = joblib.load("columns_studentStress.pkl")

# Collect user input via Streamlit widgets (sliders, selectbox)
# ...

if st.button("Predict"):
    # Build a single-row DataFrame from user input
    raw_input = {...}
    input_df = pd.DataFrame([raw_input])

    # Ensure all expected columns exist, fill missing ones with 0
    for col in expected_columns:
        if col not in input_df.columns:
            input_df[col] = 0

    # Reorder columns to match training order exactly
    input_df = input_df[expected_columns]

    # Scale input the same way training data was scaled
    scaled_input = scaler.transform(input_df)

    # Predict and display result
    prediction = model.predict(scaled_input)[0]
    st.error("High Stress Level") if prediction == 1 else st.success("Normal Stress Level")
```

## 🛠️ Tech Stack

- Python
- Streamlit
- scikit-learn
- pandas
- joblib

## 👤 Author

**Manish**

---

⭐ If you found this project helpful, consider giving it a star on GitHub!
