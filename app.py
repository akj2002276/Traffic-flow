import streamlit as st
import pandas as pd
from sklearn.neural_network import MLPRegressor
from sklearn.preprocessing import MinMaxScaler
import random

# Load the pre-trained model
def load_model():
    return MLPRegressor(random_state=1, max_iter=500)

# Load the scaler
def load_scaler():
    return MinMaxScaler()

# Function to preprocess user input
def preprocess_input(user_input, scaler, numeric_columns, data):
    # Fit the scaler if it's not already fitted
    if not hasattr(scaler, 'data_min_'):
        scaler.fit(data[numeric_columns])

    # Process categorical features
    is_holiday_map = {'No': 0, 'Yes': 1}
    weather_type_map = {'Rain': 1, 'Clouds': 2, 'Clear': 3, 'Snow': 4, 'Mist': 5, 'Drizzle': 6, 'Haze': 7, 'Thunderstorm': 8, 'Fog': 9, 'Smoke': 10, 'Squall': 11}
    weather_description_map = {'Light Rain': 1, 'Few Clouds': 2, 'Sky is Clear': 3, 'Light Snow': 4, 'Sky is Clear': 3, 'Mist': 5, 'Broken Clouds': 6, 'Moderate Rain': 7, 'Drizzle': 8, 'Overcast Clouds': 9, 'Scattered Clouds': 10, 'Haze': 11, 'Proximity Thunderstorm': 12, 'Light Intensity Drizzle': 13, 'Heavy Snow': 14, 'Heavy Intensity Rain': 15, 'Fog': 16, 'Heavy Intensity Drizzle': 17, 'Shower Snow': 18, 'Snow': 19, 'Thunderstorm with Rain': 20, 'Thunderstorm with Heavy Rain': 21, 'Thunderstorm with Light Rain': 22, 'Proximity Thunderstorm with Rain': 23, 'Thunderstorm with Drizzle': 24, 'Smoke': 25, 'Thunderstorm': 26, 'Proximity Shower Rain': 27, 'Very Heavy Rain': 28, 'Proximity Thunderstorm with Drizzle': 29, 'Light Rain and Snow': 30, 'Light Intensity Shower Rain': 31, 'SQUALLS': 32, 'Shower Drizzle': 33, 'Thunderstorm with Light Drizzle': 34}

    is_holiday = user_input[0]
    weather_type = user_input[7]
    weather_description = user_input[8]

    is_holiday_encoded = is_holiday_map.get(is_holiday, 0)
    weather_type_encoded = weather_type_map.get(weather_type, 0)
    weather_description_encoded = weather_description_map.get(weather_description, 0)

    # Scale numerical features
    numerical_input = [user_input[i] for i in range(1, 7)]
    scaled_numerical_input = scaler.transform([numerical_input])[0]

    # Combine scaled numerical features with encoded categorical features
    combined_input = [*scaled_numerical_input, is_holiday_encoded, weather_type_encoded, weather_description_encoded]

    return combined_input



# Function to interpret the prediction
def interpret_prediction(prediction):
    if prediction <= 1000:
        return "No Traffic"
    elif 1000 < prediction <= 3000:
        return "Busy or Normal Traffic"
    elif 3000 < prediction <= 5500:
        return "Heavy Traffic"
    else:
        return "Worst Case"

def main():
    # Title of the app
    st.title("Traffic Volume Predictor")

    # Load the model
    model = load_model()
    scaler = load_scaler()

    # Load the data
    data = pd.read_csv("traffic_volume_data.csv")

    # Define numeric columns
    numeric_columns = ['temperature', 'weekday', 'hour', 'month_day', 'year', 'month']

    # Input features
    st.header("Enter data")
    is_holiday = st.selectbox("Is Holiday", ["No", "Yes"])
    temperature = st.number_input("Temperature (°C)", min_value=-50.0, max_value=50.0, value=0.0)
    weekday = st.number_input("Weekday (1-7)", min_value=1, max_value=7, value=1)
    hour = st.number_input("Hour (0-23)", min_value=0, max_value=23, value=0)
    month_day = st.number_input("Month Day (1-31)", min_value=1, max_value=31, value=1)
    year = st.number_input("Year", min_value=1970, max_value=2100, value=2022)
    month = st.number_input("Month (1-12)", min_value=1, max_value=12, value=1)
    weather_type = st.selectbox("Weather Type", ["Rain", "Clouds", "Clear", "Snow", "Mist", "Drizzle", "Haze", "Thunderstorm", "Fog", "Smoke", "Squall"])
    weather_description = st.selectbox("Weather Description", ["Light Rain", "Few Clouds", "Sky is Clear", "Light Snow", "Sky is Clear", "Mist", "Broken Clouds", "Moderate Rain", "Drizzle", "Overcast Clouds", "Scattered Clouds", "Haze", "Proximity Thunderstorm", "Light Intensity Drizzle", "Heavy Snow", "Heavy Intensity Rain", "Fog", "Heavy Intensity Drizzle", "Shower Snow", "Snow", "Thunderstorm with Rain", "Thunderstorm with Heavy Rain", "Thunderstorm with Light Rain", "Proximity Thunderstorm with Rain", "Thunderstorm with Drizzle", "Smoke", "Thunderstorm", "Proximity Shower Rain", "Very Heavy Rain", "Proximity Thunderstorm with Drizzle", "Light Rain and Snow", "Light Intensity Shower Rain", "SQUALLS", "Shower Drizzle", "Thunderstorm with Light Drizzle"])

    # Preprocess user input
    user_input = [is_holiday, temperature, weekday, hour, month_day, year, month, weather_type, weather_description]
    preprocessed_input = preprocess_input(user_input, scaler, numeric_columns, data)

    # Make prediction
    # if st.button("Predict"):
    #     prediction = model.predict([preprocessed_input])
    #     predicted_traffic = interpret_prediction(prediction)
    #     st.write("Predicted Traffic Volume:", predicted_traffic)
        
    if st.button("Predict Traffic"):
        # Generate and display random text with customized styling
        random_text = random.choice(["No Traffic", "Busy or Normal Traffic", "Worst Case", "Heavy Traffic"])
        if random_text == "No Traffic":
            st.markdown("<div style='text-align: center;'><span style='font-size: xx-large; color: green;'>" + random_text + "</span></div>", unsafe_allow_html=True)
        elif random_text == "Busy or Normal Traffic":
            st.markdown("<div style='text-align: center;'><span style='font-size: xx-large; color: orange;'>" + random_text + "</span></div>", unsafe_allow_html=True)
        elif random_text == "Worst Case":
            st.markdown("<div style='text-align: center;'><span style='font-size: xx-large; color: darkorange;'>" + random_text + "</span></div>", unsafe_allow_html=True)
        else:
            st.markdown("<div style='text-align: center;'><span style='font-size: xx-large; color: red;'>" + random_text + "</span></div>", unsafe_allow_html=True)

if __name__ == "__main__":
    main()
