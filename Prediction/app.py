import streamlit as st
import pandas as pd
import pickle
from scipy import stats
import os
import subprocess

# Function to remove outliers based on z-score
def remove_outliers(df, z_thresh=3):
    return df[(abs(stats.zscore(df.select_dtypes(include=[float, int]))) < z_thresh).all(axis=1)]

# Function to preprocess data: remove outliers and handle missing values
def preprocess_data(df, handle_missing='mean'):
    # Remove outliers
    df_cleaned = remove_outliers(df)
    
    if handle_missing == 'mean':
        # Fill missing values with column mean
        df_filled = df_cleaned.fillna(df_cleaned.mean())
    elif handle_missing == 'drop':
        # Drop rows with missing values
        df_filled = df_cleaned.dropna()
    
    return df_filled

# Set up the Streamlit app title
st.title('Nostril Predictor')

# Load machine learning models from pickled files
model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))

# File uploader for data.csv
uploaded_file = st.file_uploader("Upload your raw data.csv file", type=["csv"])

if uploaded_file is not None:
    # Read the uploaded CSV file
    raw_data = pd.read_csv(uploaded_file)
    
    # Preprocess the data: always drop rows with missing values
    data = raw_data.dropna()

    # Ensure all necessary columns are numeric
    for column in ['Sensor1_Temp', 'Sensor1_Pressure', 'Sensor1_Humidity', 'Sensor2_Temp', 'Sensor2_Pressure', 'Sensor2_Humidity']:
        data.loc[:, column] = pd.to_numeric(data[column], errors='coerce')

    # Calculate average values for the columns
    avg_values = data.mean()

    # User inputs for the prediction
    temp = st.text_input('Enter the Right temperature: ', value=str(avg_values['Sensor1_Temp']))
    press = st.text_input('Enter the Right Pressure: ', value=str(avg_values['Sensor1_Pressure']))
    hum = st.text_input('Enter the Right Humidity: ', value=str(avg_values['Sensor1_Humidity']))
    temp1 = st.text_input('Enter the Left temperature: ', value=str(avg_values['Sensor2_Temp']))
    press1 = st.text_input('Enter the Left Pressure: ', value=str(avg_values['Sensor2_Pressure']))
    hum1 = st.text_input('Enter the Left Humidity: ', value=str(avg_values['Sensor2_Humidity']))

    # Button to trigger prediction
    button = st.button("Click for prediction")

    if button:
        # Validate inputs
        if temp and press and hum and temp1 and press1 and hum1:
            try:
                temp = float(temp)
                press = float(press)
                hum = float(hum)
                temp1 = float(temp1)
                press1 = float(press1)
                hum1 = float(hum1)
            except ValueError:
                st.write("Please enter valid numeric values for all inputs.")
                st.stop()

            # Create DataFrame for input values with updated column names
            new_input = pd.DataFrame({
                'Right_Temperature': [temp],
                'Right_Pressure': [press],
                'Right_Humidity': [hum],
                'Left_Temperature': [temp1],
                'Left_Pressure': [press1],
                'Left_Humidity': [hum1]
            })

            # Make predictions using the models
            prediction = model.predict(new_input)
            prediction_proba = model.predict_proba(new_input)

            prediction2 = model2.predict(new_input)
            prediction_proba2 = model2.predict_proba(new_input)

            # Format probabilities as percentages
            labels = model.classes_
            proba_percent = prediction_proba[0] * 100
            proba_dict = {label: f"{prob:.2f}%" for label, prob in zip(labels, proba_percent)}

            labels2 = model2.classes_
            proba_percent2 = prediction_proba2[0] * 100
            proba_dict2 = {label: f"{prob:.2f}%" for label, prob in zip(labels2, proba_percent2)}

            # Display the dominant nostril based on the prediction
            if prediction == 'R':
                st.markdown("<h2 style='text-align: center; font-weight: bold; color: lightgreen;'>Right Nostril is Dominant</h2>", unsafe_allow_html=True)
            elif prediction == 'L':
                st.markdown("<h2 style='text-align: center; font-weight: bold; color: lightgreen;'>Left Nostril is Dominant</h2>", unsafe_allow_html=True)
            else:
                st.write('Check the values entered')

            st.write('Prediction Probabilities (Dominant Nostril):', proba_dict)

            # Determine the activity based on the second prediction
            activity = None
            if prediction2 == 'W':
                activity = "Walking"
            elif prediction2 == 'S':
                activity = "Sitting"
            elif prediction2 == 'D':
                activity = "Deep Breathing"
            else:
                st.write('Check the values entered')

            if activity:
                st.markdown(f"<h2 style='text-align: center; font-weight: bold; color: lightgreen;'>The person is {activity}</h2>", unsafe_allow_html=True)
                st.write('Prediction Probabilities (Activity):', proba_dict2)

            file_name = uploaded_file.name
            os.system(f"python main.py {file_name}")
            
            # Check if Results.csv is generated
            if os.path.exists('Results.csv'):
                # Read Results.csv generated by main.py
                results = pd.read_csv('Results.csv')

                # Extract values from the last row of Results.csv
                last_row = results.iloc[-1]

                # Display the results for the right nostril
                st.markdown("<h2 style='font-weight: bold;'>Right Nostril Data</h2>", unsafe_allow_html=True)
                st.write('Temperature:', temp)
                st.write('Humidity:', hum)
                st.write('Pressure:', press)
                st.write('Breathing Rate:', last_row['Breathing Rate 1'])
                st.write('Inhale Interbreath Interval:', last_row['Inhale Interbreath Interval 1'])
                st.write('Exhale Interbreath Interval:', last_row['Exhale Interbreath Interval 1'])
                st.write('Interbreath Interval:', last_row['Interbreath Interval 1'])
                st.write('Tidal Volume Rate:', last_row['Tidal Volume Rate 1'])
                st.write('Minute Ventilation:', last_row['Minute Ventilation 1'])

                # Display the results for the left nostril
                st.markdown("<h2 style='font-weight: bold;'>Left Nostril Data</h2>", unsafe_allow_html=True)
                st.write('Temperature:', temp1)
                st.write('Humidity:', hum1)
                st.write('Pressure:', press1)
                st.write('Breathing Rate:', last_row['Breathing Rate 2'])
                st.write('Inhale Interbreath Interval:', last_row['Inhale Interbreath Interval 2'])
                st.write('Exhale Interbreath Interval:', last_row['Exhale Interbreath Interval 2'])
                st.write('Interbreath Interval:', last_row['Interbreath Interval 2'])
                st.write('Tidal Volume Rate:', last_row['Tidal Volume Rate 2'])
                st.write('Minute Ventilation:', last_row['Minute Ventilation 2'])
            else:
                st.write("Results.csv not found. Ensure main.py runs correctly and generates Results.csv.")

    else:
        st.write("Please enter values for all inputs.")

else:
    st.write("Please upload a data.csv file to proceed.")
