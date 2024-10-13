from fastapi import FastAPI
from pydantic import BaseModel
import pickle
import numpy as np
app = FastAPI()

@app.get("/")
def read_root():
    return {"message": "Welcome to the Weather Prediction API!"}

# Other API routes can go here...

# Load the trained model and encoders
with open('weather_model.pkl', 'rb') as f:
    model = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)

with open('label_encoder.pkl', 'rb') as f:
    label_encoder = pickle.load(f)

# Define the input data structure
class WeatherInput(BaseModel):
    temperature: float
    humidity: float
    wind_speed: float
    # Add other relevant features

@app.post("/predict")
def predict_weather(data: WeatherInput):
    input_features = np.array([[data.temperature, data.humidity, data.wind_speed]])
    scaled_features = scaler.transform(input_features)
    prediction = model.predict(scaled_features)
    weather = label_encoder.inverse_transform(prediction)
    return {"weather": weather[0]}

