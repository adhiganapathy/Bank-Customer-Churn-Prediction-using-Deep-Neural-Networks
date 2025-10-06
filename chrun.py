import streamlit as st
import pickle
import os
import pandas as pd
from tensorflow import keras
import keras
from keras.models import load_model

# get the directory where the script is located
BASE_DIR = os.path.dirname(os.path.abspath(__file__))

# build path relative to the script
csv_path = os.path.join(BASE_DIR, "Churn_Modelling.csv")

# read the CSV
data = pd.read_csv(csv_path)


# -------------------------------
# Load preprocessed data
# -------------------------------

pickle_path = os.path.join(BASE_DIR, "PickleFile.pkl")
with open(pickle_path, "rb") as file:
     sc= pickle.load(file)
     ac= pickle.load(file)

model_path = os.path.join(BASE_DIR, "Model.keras")


classifier = keras.models.load_model(model_path,safe_mode=False)

st.sidebar.title("Bank Customer Churn Prediction")  




def predictions():
     


    CN = st.text_input("Customer Name") 
    Geography = st.selectbox("Geography", ["France", "Spain", "Germany"])
    Gender = st.selectbox("Gender", ["Male", "Female"])
    Age = st.number_input("Age", min_value=18)
    Tenure = st.number_input("Tenure (Years)")
    Balance = st.number_input("Balance")
    NumOfProducts = st.number_input("Number of Products")
    CS = st.number_input("Credit Score")
    HasCrCard = st.selectbox("Has Credit Card", ["Yes", "No"])
    IsActiveMember = st.selectbox("Is Active Member", ["Yes", "No"])
    EstimatedSalary = st.number_input("Estimated Salary")

    map = {"Male":1,"Female":0}
    Gender= map[Gender]

    map = {"Yes":1,"No":0}
    HasCrCard= map[HasCrCard]

    map = {"Yes":1,"No":0}
    IsActiveMember= map[IsActiveMember]

    map = {"France":0,"Spain":2,"Germany":1}
    Geography= map[Geography]

    input_data = [[CS, Age, Tenure, Balance, NumOfProducts, HasCrCard, IsActiveMember, EstimatedSalary, Geography, Gender]]

    Features = sc.transform(input_data)


    if st.button("Recommend"):
        prediction =classifier.predict(Features)
        predicted_label = (prediction > 0.5).astype(int)
        print(predicted_label[0])
        if predicted_label[0] == 1:
            st.error(f"Customer {CN}  is likely to stay with Justice Bank.")
        else:
            st.success("Customer {} is likely to churn from Justice Bank.".format(CN))
         


page =st.sidebar.radio("Go To",["Overview","Model Evaluation","Prediction"])

def main():

    if page == "Overview":
        st.title("📊 Dataset Overview")
 
        st.write(data.head())

    elif page == "Model Evaluation":
        st.title("📈 Model Evaluation")
    

    # Display in Streamlit
 
        st.write("Accuracy Score :", f'{ac:.2%}')

    

    elif page=="Prediction":
    
        st.title( " Prediction")
        predictions()



if __name__ == "__main__":
    main()         
