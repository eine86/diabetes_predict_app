import numpy as np
import pickle
import streamlit as st

# load model
model = pickle.load(open("C:/Users/false/OneDrive/Documents/DataLearning/project - app -diabetes/trained_model.sav", "rb"))

# create prediction function
def predict(input_data:list):
    # preprocessing
    input_arr = np.asarray(input_data)
    conv_arr = input_arr.reshape(1, -1)

    # predict
    prediction = model.predict(conv_arr)
    print(prediction)

    if prediction[0] == 0:
        return "This person has no diabetes"
    else:
        return "This person has diabetes"


# create streamlit app
def main():
    # title
    st.title("Diabetes Prediction Web App")

    # user input data
    Pregnancies = st.text_input("Pregnancies")
    Glucose = st.text_input("Glucose Level")
    BloodPressure = st.text_input("Blood Pressure Level")
    SkinThickness = st.text_input("Skin Thickness")
    Insulin = st.text_input("Insulin")
    BMI = st.text_input("BMI")
    DPF = st.text_input("Diabetes Pedigree Function")
    Age = st.text_input("Age")


    outcome = "" # empty string, since button is not pressed yet
    # button
    if st.button("Predict"):
        outcome = predict([Pregnancies, Glucose, BloodPressure, SkinThickness, Insulin, BMI, DPF, Age])
    
    st.success(outcome)



# runs only in terminal
if __name__ == "__main__":
    main()