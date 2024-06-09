import streamlit as st
import numpy as np
import pickle
import pandas as pd

def get_clean_data():
    data = pd.read_csv("diabetes.csv")
    return data

def add_sidebar():
    st.sidebar.header("Diabetes Symptom Measurements")

    data = get_clean_data()
    slider_label = [
        ("Pregnancies", "Pregnancies"),
        ("Glucose", "Glucose"),
        ("BloodPressure", "BloodPressure"),
        ("SkinThickness", "SkinThickness"),
        ("Insulin", "Insulin"),
        ("BMI", "BMI"),
        ("DiabetesPedigreeFunction", "DiabetesPedigreeFunction"),
        ("Age", "Age")
    ]

    input_dict = {}
    for label, key in slider_label:
        input_dict[key] = st.sidebar.slider(
            label,
            min_value=float(0),
            max_value=float(data[key].max()),
            value=float(data[key].mean())
        )
    return input_dict

def get_scaled_values(input_dict, scaler):
    input_array = np.array(list(input_dict.values())).reshape(1, -1)
    scaled_array = scaler.transform(input_array)
    return scaled_array

def add_prediction(input_data):
    with open("model.pkl", "rb") as model_in:
        model = pickle.load(model_in)
    with open("scaler.pkl", "rb") as scaler_in:
        scaler = pickle.load(scaler_in)
    
    input_scaled = get_scaled_values(input_data, scaler)
    
    prediction = model.predict(input_scaled)

    if prediction[0] == 0:
        st.write("Diabetes not detected")
    else:
        st.write("Diabetes detected")

    probabilities = model.predict_proba(input_scaled)[0]
    st.write(f"The probability of not having diabetes is: {probabilities[0]:.2f}")
    st.write(f"The probability of having diabetes is: {probabilities[1]:.2f}")
    st.write("This app can assist medical professionals in making a diagnosis, but should not be used as a substitute for professional diagnosis")

def main():
    st.set_page_config(page_title="Patient Diabetes Prediction", page_icon=":doctor:", layout="centered", initial_sidebar_state="expanded")
    
    input_data = add_sidebar()
    
    st.title("Patient Diabetes Prediction")
    st.write("Please use this app to predict the likelihood of diabetes based on several health metrics.")
    
    if st.button("Predict"):
        add_prediction(input_data)
    
    if st.button("About"):
        st.text("This app is built with Streamlit")
        st.text("It helps in predicting diabetes based on user inputs")

if __name__ == "__main__":
    main()