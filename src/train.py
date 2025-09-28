import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import r2_score
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
import joblib
import os

print("Starting model training...")
os.makedirs('models', exist_ok=True)

# 1. Load Data
try:
    df = pd.read_csv('data/crop_yield.csv')
    print("Dataset loaded successfully.")
except FileNotFoundError:
    print("Error: Dataset file not found. Make sure you run 'dvc pull'.")
    exit()

# 2. Preprocessing and Feature Selection
df = df.dropna()

# Define the target and the features to be used for prediction
target = 'Yield'
# We use factors that influence yield, NOT the ones that are part of its calculation (like Production)
features = ['Crop', 'Crop_Year', 'Season', 'State', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

X = df[features]
y = df[target]

# Define which columns are categorical and which are numerical
categorical_features = ['Crop', 'Season', 'State']
numerical_features = ['Crop_Year', 'Area', 'Annual_Rainfall', 'Fertilizer', 'Pesticide']

# 3. Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print("Data split complete.")

# 4. Create a Preprocessing and Modeling Pipeline
# This pipeline will handle categorical data and then train the model
preprocessor = ColumnTransformer(
    transformers=[
        ('num', 'passthrough', numerical_features),
        ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
    ])

# Define the model
model = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)

# Bundle preprocessing and the model into a single pipeline
pipeline = Pipeline(steps=[('preprocessor', preprocessor),
                           ('regressor', model)])

# 5. Train the Pipeline
print("Training the model pipeline...")
pipeline.fit(X_train, y_train)
print("Model training complete.")

# 6. Evaluate the Pipeline
y_pred = pipeline.predict(X_test)
score = r2_score(y_test, y_pred)
print(f"Model R^2 Score: {score:.4f}")

# 7. Save the entire pipeline (preprocessor + model)
model_path = 'models/crop_yield_pipeline.pkl'
joblib.dump(pipeline, model_path)
print(f"Model pipeline saved to {model_path}")