import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os
import yaml
import json
import numpy as np

print("Starting model training...")
os.makedirs('models', exist_ok=True)

with open('params.yaml', 'r') as f:
    params = yaml.safe_load(f)

test_size = params['data_ingestion']['test_size']
model_params = params['model_building']

df = pd.read_csv('data/crop_yield.csv').dropna()
print("Dataset loaded successfully.")

target = 'Yield'
features = ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']
X = df[features]
y = df[target]

categorical_features = ['Crop', 'Season', 'State']
numerical_features = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=model_params['random_state'])
print(f"Data split with test_size={test_size}.")

preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

model = RandomForestRegressor(
    n_estimators=model_params['n_estimators'],
    max_features=model_params['max_features'],
    random_state=model_params['random_state'],
    n_jobs=-1
)

pipeline = Pipeline(steps=[('preprocessor', preprocessor), ('regressor', model)])

print("Training the model pipeline...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

y_pred = pipeline.predict(X_test)
r2 = r2_score(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"R-squared (R²): {r2:.4f}, MAE: {mae:.4f}, RMSE: {rmse:.4f}")

joblib.dump(pipeline, 'models/crop_yield_pipeline.pkl')
print("Model pipeline saved.")

with open('metrics.json', 'w') as f:
    json.dump({'r2_score': r2, 'mae': mae, 'mse': mse, 'rmse': rmse}, f, indent=4)
print("Metrics saved to metrics.json")