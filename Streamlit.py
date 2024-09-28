import streamlit as st
import pickle
import numpy as np

with open(r"C:\Users\jayas\Vs Code\LLM_Model.pkl", "rb") as file:
    model = pickle.load(file)

st.title("Salary Prediction App")
st.write("Thia will predict your salary based on your experiance")

years_input = st.number_input("Enter your Expereance",min_value = 0.0,max_value = 50.0,value = 1.0,step = 0.5)

if st.button('predict salary'):
    experiance_input = np.array([[years_input]])
    prediction = model.predict(experiance_input)
    
    st.success(f"The predicted exp is {years_input} is ${prediction[0]:,.2f}")
