import streamlit as st
import joblib
import numpy as np

# Load the MLP model
mlp_model = joblib.load('mlp_model.joblib')

st.title('Flower Species Prediction')
st.write('Enter the measurements of the flower to predict its species.')

# Create input fields for features
sepal_length = st.number_input('Sepal Length', min_value=0.0, max_value=10.0, value=5.0, step=0.1)
sepal_width = st.number_input('Sepal Width', min_value=0.0, max_value=10.0, value=3.0, step=0.1)
petal_length = st.number_input('Petal Length', min_value=0.0, max_value=10.0, value=4.0, step=0.1)
petal_width = st.number_input('Petal Width', min_value=0.0, max_value=10.0, value=1.5, step=0.1)

if st.button('Predict Species'):
    # Prepare the input features as a NumPy array
    # Ensure the order matches the model's training: sepal length, sepal width, petal length, petal width
    features = np.array([[sepal_length, sepal_width, petal_length, petal_width]])

    # Make prediction
    prediction = mlp_model.predict(features)

    st.success(f'The predicted species is: {prediction[0]}')
