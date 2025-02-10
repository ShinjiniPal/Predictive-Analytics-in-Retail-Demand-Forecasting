from fastapi import FastAPI
import joblib
import numpy as np
from pydantic import BaseModel

app = FastAPI()

# Load the trained model and preprocessing tools
model = joblib.load("retail_demand_model.pkl")
ohe = joblib.load("ohe_encoder.pkl")
scaler = joblib.load("scaler.pkl")

# Define request schema
class PredictionRequest(BaseModel):
    numerical_features: list[float]
    categorical_features: list[str]

@app.post("/predict")
async def predict(request: PredictionRequest):
    try:
        # Extract features and preprocess
        numerical_features = np.array(request.numerical_features).reshape(1, -1)
        categorical_features = np.array(request.categorical_features).reshape(1, -1)

        # Encode categorical features
        categorical_encoded = ohe.transform(categorical_features)

        # Ensure numerical features match the expected size
        if numerical_features.shape[1] != scaler.n_features_in_:
            return {"error": f"Expected {scaler.n_features_in_} numerical features, but got {numerical_features.shape[1]}"}

        # Scale numerical features
        numerical_scaled = scaler.transform(numerical_features)

        # Combine features
        features = np.hstack((numerical_scaled, categorical_encoded))

        # Make prediction
        prediction = model.predict(features)
        return {"prediction": prediction.tolist()}

    except Exception as e:
        return {"error": str(e)}
        
if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=5001)