import pandas as pd
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

# Load the dataset
data = pd.read_csv('/home/samyak/weather-prediction-system/1.csv')

# Drop rows with missing values
data = data.dropna()

# Define X (features) and y (target)
X = data[['precipitation', 'temp_max', 'temp_min', 'wind']]  # Features
y = data['weather']  # Target (the weather column)

# Encode categorical target variable (weather) using LabelEncoder
label_encoder = LabelEncoder()
y = label_encoder.fit_transform(y)

# Feature scaling: Standardize the features to bring them onto the same scale
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X_scaled, y, test_size=0.2, random_state=42)

# Save the scaler and label encoder to use during model prediction
with open('scaler.pkl', 'wb') as f:
    pickle.dump(scaler, f)

with open('label_encoder.pkl', 'wb') as f:
    pickle.dump(label_encoder, f)

# You can save the preprocessed data too (optional)
with open('preprocessed_data.pkl', 'wb') as f:
    pickle.dump((X_train, X_test, y_train, y_test), f)

print("Data preprocessing completed and saved.")

