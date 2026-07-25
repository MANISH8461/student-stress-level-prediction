# import streamlit as st
# import pandas as pd
# import joblib

# model = joblib.load("logisticReg_Student_Stress_Level.pkl")
# scaler = joblib.load("scaler_studentLogistic_.pkl")
# expected_columns = joblib.load("columns_studentStress.pkl")


# st.title("Student Stress Prediction by Manish")
# st.markdown("Provide the following details.")

# studentType = st.selectbox("Student Type", ["School", "College", "Working Student"])
# sleepHours = st.slider("Sleep Hours", 2.02, 9.98)
# studyHours = st.slider("Study Hours", -1.43, 21.97)
# socialMediaHours = st.slider("Social Media Hours", 3.99, 9.96)
# attendance = st.slider("Attendance", -5.0, 120.0)
# examPressure = st.slider("Exam Pressure", 1, 10)
# familySupport = st.slider("Family Support", 1.0, 10.0, 1.0)
# month = st.slider("Month", 1, 12,1)



# if st.button("Predict"):
#     raw_input = {
#         'Student Type' +  studentType : 1,
#         "Sleep Hours": sleepHours,
#         'Study Hours': studyHours,
#         'Social Media Hours': socialMediaHours,
#         'Attendance': attendance,
#         'Exam Pressure': examPressure,
#         'Family Support': familySupport,
#         'Month': month,
#     }

#     input_df = pd.DataFrame([raw_input])

#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     input_df = input_df[expected_columns]

#     scaled_input = scaler.transfom(input_df)
#     prediction = model.prediction(scaled_input)[0]

#     if prediction == 1:
#         st.error("High Stress Level")
#     else:
#         st.success("Normal Stress Level")







import streamlit as st
import pandas as pd
import joblib

model = joblib.load("logisticReg_Student_Stress_Level.pkl")
scaler = joblib.load("scaler_studentLogistic_.pkl")
expected_columns = joblib.load("columns_studentStress.pkl")


st.title("Student Stress Prediction by Manish")
st.markdown("Provide the following details.")

studentType = st.selectbox("Student Type", ["School", "College", "Working Student"])
sleepHours = st.slider("Sleep Hours", 2.02, 9.98)
studyHours = st.slider("Study Hours", -1.43, 21.97)
socialMediaHours = st.slider("Social Media Hours", 3.99, 9.96)
attendance = st.slider("Attendance", -5.0, 120.0)
examPressure = st.slider("Exam Pressure", 1, 10)
familySupport = st.slider("Family Support", 1.0, 10.0, 1.0)
month = st.slider("Month", 1, 12, 1)


# if st.button("Predict"):
#     raw_input = {
#         "Sleep Hours": sleepHours,
#         "Study Hours": studyHours,
#         "Social Media Hours": socialMediaHours,
#         "Attendance": attendance,
#         "Exam Pressure": examPressure,
#         "Family Support": familySupport,
#         "Month": month,
#         "Student Type_" + studentType: 1,   # fixed: underscore separator to match training column names
#     }

#     input_df = pd.DataFrame([raw_input])

#     for col in expected_columns:
#         if col not in input_df.columns:
#             input_df[col] = 0

#     input_df = input_df[expected_columns]

#     scaled_input = scaler.transform(input_df)          # fixed: transfom -> transform
#     prediction = model.predict(scaled_input)[0]          # fixed: prediction -> predict

#     if prediction == 1:
#         st.error("High Stress Level")
#     else:
#         st.success("Normal Stress Level")

if st.button("Predict"):
    student_type_map = {"School": 0, "College": 1, "Working Student": 2}  # verify order from training notebook

    raw_input = {
        "Student_Type": student_type_map[studentType],
        "Sleep_Hours": sleepHours,
        "Study_Hours": studyHours,
        "Social_Media_Hours": socialMediaHours,
        "Attendance": attendance,
        "Exam_Pressure": examPressure,
        "Family_Support": familySupport,
        "Month": month,
    }

    input_df = pd.DataFrame([raw_input])
    input_df = input_df[expected_columns]

    scaled_input = scaler.transform(input_df)
    prediction = model.predict(scaled_input)[0]

    if prediction == 1:
        st.error("High Stress Level")
    else:
        st.success("Normal Stress Level")