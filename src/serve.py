from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os
import boto3

app = FastAPI()

# Path configuration
MODEL_PATH = os.environ.get("MODEL_PATH", os.path.expanduser("~/models/model.pkl"))

def download_model():
    """Tải file model.pkl từ S3 về máy khi server khởi động."""
    if os.environ.get("MOCK_DEPLOY") == "true":
        print("Mock deploy active: skipping S3 download.")
        return
    bucket = os.environ.get("CLOUD_BUCKET")
    if not bucket:
        print("Warning: CLOUD_BUCKET environment variable not set. Skipping download.")
        return
    s3_key = "models/latest/model.pkl"
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)
    print(f"Downloading model from S3 bucket: {bucket}, key: {s3_key} to {MODEL_PATH}")
    s3 = boto3.client("s3")
    s3.download_file(bucket, s3_key, MODEL_PATH)
    print("Model downloaded successfully!")

# Download model from S3 when starting (unless run in a test or mock setting without bucket)
try:
    download_model()
except Exception as e:
    print(f"Failed to download model: {e}")

# If the file doesn't exist at MODEL_PATH, fallback to local path for development/testing
if not os.path.exists(MODEL_PATH) and os.path.exists("models/model.pkl"):
    MODEL_PATH = "models/model.pkl"

model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model loaded successfully from {MODEL_PATH}")
else:
    print(f"Warning: Model not found at {MODEL_PATH}")

class PredictRequest(BaseModel):
    features: list[float]

@app.get("/health")
def health():
    return {"status": "ok"}

@app.post("/predict")
def predict(req: PredictRequest):
    if model is None:
        raise HTTPException(status_code=503, detail="Model is not loaded.")
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail=f"Expected 12 features, got {len(req.features)}")
    pred = model.predict([req.features])
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    prediction = int(pred[0])
    return {"prediction": prediction, "label": label_map[prediction]}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
