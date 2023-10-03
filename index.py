import serial
import numpy as np
import tensorflow as tf
import requests
import time

API_KEY = "2f975c49958d44e09d2db874128ace94"
CITY = "Zugdidi"

# Function to predict water needed based on soil moisture
def predict_water_needed(soil_moisture, model):
    soil_moisture = np.array([soil_moisture]).reshape(-1, 1)
    water_needed = model.predict(soil_moisture)
    return water_needed[0][0]

def get_weather_data():
    url = f"https://api.openweathermap.org/data/2.5/weather?q={CITY}&appid={API_KEY}"
    response = requests.get(url)

    if response.status_code == 200:
        weather_data = response.json()

        main_weather = weather_data["weather"][0]["main"]
        description = weather_data["weather"][0]["description"]
        temperature_kelvin = weather_data["main"]["temp"]
        temperature_celsius = temperature_kelvin - 273.15
        humidity = weather_data["main"]["humidity"]
        wind_speed = weather_data["wind"]["speed"]

        print(f"Weather in {CITY}:")
        print(f"Main Weather: {main_weather}")
        print(f"Description: {description}")
        print(f"Temperature: {temperature_celsius:.2f}°C")
        print(f"Humidity: {humidity}%")
        print(f"Wind Speed: {wind_speed} m/s")
    else:
        print("Error getting the weather data")

# Create a simple linear regression model (replace this with your trained model)
model = tf.keras.Sequential([
    tf.keras.layers.Input(shape=(1,)),
    tf.keras.layers.Dense(1)
])

model.compile(optimizer='adam', loss='mse')
# Dummy training data for the sake of example, replace with actual training
X_train = np.array([10, 20, 30, 40, 50])
y_train = np.array([5, 10, 15, 20, 25])
model.fit(X_train, y_train, epochs=100, verbose=0)

# Read soil moisture data from Arduino
#ser = serial.Serial('/dev/ttyACM8', 9600)  # Change to your Arduino's port

while True:
    try:
        soil_moisture = float(ser.readline().decode().strip())  # Read and decode data from Arduino
        water_needed = predict_water_needed(soil_moisture, model)  # Predict water needed
        print(f"Soil Moisture: {soil_moisture} - Water Needed for 1000m^2: {water_needed} liters")
    except Exception as e:
        print(e)
        print("success")
        get_weather_data()  # Corrected this line to call the function
        break
