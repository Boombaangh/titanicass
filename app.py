import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Set up Streamlit page layout
st.set_page_config(layout="wide")
st.title("🚢 Titanic Survival Predictor")

# Load trained model
survival_model = pickle.load(open('titanic_survival_model.pkl', 'rb'))

# User input fields
ticket_class = st.selectbox("Passenger Class (Pclass)", [1, 2, 3])
passenger_gender = st.selectbox("Sex", ["Male", "Female"])
age_input = st.number_input("Age", min_value=0, max_value=100, step=1)
sibling_spouse_count = st.number_input("Number of Siblings/Spouses Aboard (SibSp)", min_value=0, step=1)
parent_child_count = st.number_input("Number of Parents/Children Aboard (Parch)", min_value=0, step=1)
fare_price = st.number_input("Fare", min_value=0.0, step=0.1)
embark_point = st.selectbox("Embarked", ["C", "Q", "S"])

# Convert categorical inputs to match model's training format
is_male = 1 if passenger_gender == "Male" else 0  # Model was trained with 'Sex_male'
embarked_Q_flag = 1 if embark_point == "Q" else 0
embarked_S_flag = 1 if embark_point == "S" else 0

# Organize user input in the correct format
user_features = {
    'Pclass': ticket_class,
    'Age': age_input,
    'SibSp': sibling_spouse_count,
    'Parch': parent_child_count,
    'Fare': fare_price,
    'Sex_male': is_male,
    'Embarked_Q': embarked_Q_flag,
    'Embarked_S': embarked_S_flag
}

# Convert to DataFrame with correct structure
input_dataframe = pd.DataFrame(user_features, index=[0])

# Predict button
if st.button("Check Survival"):
    survival_prediction = survival_model.predict(input_dataframe)

    # Display prediction outcome
    survival_output = "Survived 🟢" if survival_prediction[0] == 1 else "Did Not Survive 🔴"
    st.success(f"Prediction: **{survival_output}**")
