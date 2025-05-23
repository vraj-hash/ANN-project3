import streamlit as st
import numpy as np
import tensorflow as tf
from sklearn.preprocessing import StandardScaler,LabelEncoder,OneHotEncoder
import pandas as pd
import pickle


#load the trained model
model=tf.keras.models.load_model('model.h5')

#load the encoders and scalers
with open('one_hot_encoder_geo.pkl','rb') as file:
    label_encoder_geo=pickle.load(file)

with open('label_encoder_gender.pkl','rb') as file:
    label_encoder_gender=pickle.load(file)

with open('scaler.pkl','rb') as file:
    scaler=pickle.load(file)

# streamlit app
st.title('Customer churn prediction')


#user input
geography=st.selectbox('Geography',label_encoder_geo.categories_[0])
gender=st.selectbox('Gender',label_encoder_gender.classes_)
age=st.slider('Age',18,92)
balance=st.number_input('Balance')
credit_score=st.number_input('Credit_score')
estimated_salary=st.number_input('Estimated Salary')
tenure=st.slider('Tenure',0,10)
num_of_products=st.slider('Number of Products',1,4)
has_cr_card=st.selectbox('Has Credit Card',[0,1])
is_active_member=st.selectbox('Is Active Member',[0,1])

#prepare input data
input_data = pd.DataFrame({
    'CreditScore':[credit_score],
    'Gender': [label_encoder_gender.transform([gender])[0]],
    'Age':[age],
    'Tenure':[tenure],
    'Balance':[balance],
    'NumOfProducts':[num_of_products],
    'HasCrCard':[has_cr_card],
    'IsActiveMember':[is_active_member],
    'EstimatedSalary':[estimated_salary]
})

#one hot encoding for geography
geo_encoded=label_encoder_geo.transform([[geography]]).toarray()
geo_encoded_df=pd.DataFrame(geo_encoded,columns=label_encoder_geo.get_feature_names_out(['Geography']))

#combining one_hot_coded data
input_data = pd.concat([input_data.reset_index(drop=True), geo_encoded_df], axis=1)

#scale the input
input_scaled=scaler.transform(input_data)

#prediction
prediction=model.predict(input_scaled)
probability=prediction[0][0]

probability=prediction[0][0]
# print(f"Churn probability is {probability}")
st.write(f"Churn probability is {probability:.2f}")
if probability > 0.5:
    st.write('Customer is likely to churn')
else:
    st.write('Customer is not likely to churn')

