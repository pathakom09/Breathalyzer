import streamlit as st
import pandas as pd
import pickle
import os

# Set up the Streamlit app title
st.title('Nostril Predictor')

# Load machine learning models from pickled files
model = pickle.load(open('model.pkl', 'rb'))
model2 = pickle.load(open('model2.pkl', 'rb'))

additional_data = pd.read_csv('S-W-D data.csv')

# User inputs for the prediction
temp = st.text_input('Enter the Right temperature: ')
press = st.text_input('Enter the Right Pressure: ')
hum = st.text_input('Enter the Right Humidity: ')
temp1 = st.text_input('Enter the Left temperature: ')
press1 = st.text_input('Enter the Left Pressure: ')
hum1 = st.text_input('Enter the Left Humidity: ')

# Button to trigger prediction
button = st.button("Click for prediction")

# Determine the dominant nostril based on environmental conditions
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
        elif prediction == 'B':
            st.markdown("<h2 style='text-align: center; font-weight: bold; color: lightgreen;'>Both Nostrils are Dominant</h2>", unsafe_allow_html=True)
        else:
            st.write('Check the values entered')

        st.write('Prediction Probabilities (Dominant Nostril):', proba_dict)

        # Determine the activity based on the second prediction
        activity = None
        if prediction2 == 'W':
            activity = "Walking"
            num_records = 14000 
        elif prediction2 == 'S':
            activity = "Sitting"
            num_records = 14000 
        elif prediction2 == 'D':
            activity = "Deep Breathing"
            num_records = 14000 
        else:
            st.write('Check the values entered')

        if activity:
            st.markdown(f"<h2 style='text-align: center; font-weight: bold; color: lightgreen;'>The person is {activity}</h2>", unsafe_allow_html=True)
            st.write('Prediction Probabilities (Activity):', proba_dict2)

            # Merge new_input with selected_additional_data along rows
            result_df = pd.concat([new_input] * num_records, ignore_index=True)

            # Sample the necessary number of records from additional_data
            sampled_additional_data = additional_data.sample(n=num_records, replace=True)

            # Concatenate new_input DataFrame with the sampled additional data along rows
            result_df = pd.concat([sampled_additional_data.reset_index(drop=True)], axis=1)

            # Filter result_df to keep only the necessary columns
            result_df = result_df[[
                'Right_Temperature', 'Right_Pressure', 'Right_Humidity',
                'Left_Temperature', 'Left_Pressure', 'Left_Humidity'
            ]]

            # Rename columns for saving to prediction.csv
            result_df = result_df.rename(columns={
                'Right_Temperature': 'Sensor1_Temp',
                'Right_Pressure': 'Sensor1_Pressure',
                'Right_Humidity': 'Sensor1_Humidity',
                'Left_Temperature': 'Sensor2_Temp',
                'Left_Pressure': 'Sensor2_Pressure',
                'Left_Humidity': 'Sensor2_Humidity'
            })

            # Remove rows with NaN or empty values
            result_df_cleaned = result_df.dropna().dropna(how='all')

            # Save cleaned result to CSV with only the necessary columns
            result_df_cleaned.to_csv('prediction.csv', index=False)

            # Run main.py using os.system when the button is clicked
            os.system("python main.py")

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
        st.write("Please enter values for all inputs.")
