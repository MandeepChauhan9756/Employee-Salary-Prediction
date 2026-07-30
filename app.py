import streamlit as st 
import pandas as pd 
import joblib

# ==========================================
# Load Trained Model
# ==========================================
model = joblib.load("model/salary_prediction_model.pkl")
preprocessor = joblib.load("model/preprocessor.pkl")

# ==========================================
# Page Configuration
# ==========================================
st.set_page_config(
    page_title = "Employee Salary Prediction",
    page_icon = "💼",
    layout = "wide"
)

st.title("💼 Employee Salary Prediction")
st.caption("Machine Learning Project | Gradient Boosting Regressor | Developed using Streamlit")
st.write("""
         Predict an employee's monthly salary using Machine Learning.
         Model Used: **Gradient Boosting Regressor**
         """)
st.divider()

# st.sidebar.header("About")
# st.sidebar.info(
#     """
#     This application predicts the monthly salary of an employee using a Machine Learning model trained on an employee dataset.
#     """
# )

st.sidebar.header("📌 Project Information")
st.sidebar.write("### Model")
st.sidebar.success("Gradient Boosting Regressor")
st.sidebar.write("### Dataset")
st.sidebar.info("Employee Salary Dataset (5000+ Records)")

st.sidebar.write("### Features Used")
st.sidebar.write("17 Features")

st.sidebar.write("### Developer")
st.sidebar.write("Mandeep Chauhan")

# ==========================================
# Employee Details Form
# ==========================================
col1, col2 = st.columns(2)
with col1:
    age = st.number_input(
        "Age",
        min_value=18,
        max_value=65,
        value=28
    )
    
    gender = st.selectbox(
        "Gender",
        ["Male", "Female"]
    )
    
    education = st.selectbox(
        "Education",
        [
            "High School",
            "Diploma",
            "Bachelor's",
            "Master's",
            "PhD"
        ]
    )
    
    department = st.selectbox(
        "Department",
        [
            "IT",
            "HR",
            "Finance",
            "Sales",
            "Marketing",
            "Operations"
        ]
    )
    
    experience = st.number_input(
        "Experience",
        min_value=0,
        max_value=40,
        value=5
    )
    
    performance = st.slider(
        "Performance Rating",
        1,
        5,
        3
    )
    
    overtime = st.selectbox(
        "Overtime",
        ["Yes", "No"]
    )

    work_mode = st.selectbox(
        "Work Mode",
        [
            "Office",
            "Hybrid",
            "Remote"
        ]
    )
    
with col2:
    city = st.text_input(
        "City",
        "Delhi"
    )

    company_size = st.selectbox(
        "Company Size",
        [
            "Startup",
            "Medium",
            "Large",
            "Enterprise"
        ]
    )

    marital_status = st.selectbox(
        "Marital Status",
        [
            "Single",
            "Married",
            "Divorced"
        ]
    )
    
    certification = st.number_input(
        "Certification Count",
        min_value=0,
        max_value=20,
        value=3
    )

    skills = st.slider(
        "Skills Score",
        40,
        100,
        80
    )

    years_company = st.number_input(
        "Years At Company",
        min_value=0,
        max_value=40,
        value=3
    )

    promotion = st.selectbox(
        "Promotion Last 5 Years",
        ["Yes", "No"]
    )

    annual_bonus = st.number_input(
        "Annual Bonus",
        min_value=0,
        value=80000
    )

    job_title = st.text_input(
        "Job Title",
        "Software Engineer"
    )
    
# ==========================================
# Predict Salary Button
# ==========================================

if st.button("Predict Monthly Salary"):
    input_df = pd.DataFrame({
        "Age": [age],
        "Gender": [gender],
        "Education": [education],
        "Department": [department],
        "Job_Title": [job_title],
        "Experience": [experience],
        "Performance_Rating": [performance],
        "Overtime": [overtime],
        "Work_Mode": [work_mode],
        "City": [city],
        "Company_Size": [company_size],
        "Marital_Status": [marital_status],
        "Certification_Count": [certification],
        "Skills_Score": [skills],
        "Years_At_Company": [years_company],
        "Promotion_Last_5_Years": [promotion],
        "Annual_Bonus": [annual_bonus]
    })
    
    # Apply preprocessing
    input_encoded = preprocessor.transform(input_df)

    # Predict salary
    prediction = model.predict(input_encoded)[0]

    # st.success(f"Predicted Monthly Salary: ₹{prediction:,.2f}")
    st.divider()
    st.subheader("Prediction Result")
    st.metric(
        label="Predicted Monthly Salary",
        value=f"₹{prediction:,.2f}"
    )
    
    st.info("Prediction generated using the Gradient Boosting Regressor model.")
    
    st.divider()

st.markdown(
    """
    <center>
        <h5>Employee Salary Prediction using Machine Learning</h5>
        <p>Developed by <b>Mandeep Chauhan</b></p>
    </center>
    """,
    unsafe_allow_html=True
)