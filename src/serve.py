from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
import joblib
import os

app = FastAPI()

MODEL_PATH = "models/model.pkl"


class PredictRequest(BaseModel):
    features: list[float]


# Load model khi server khoi dong (chi load local, khong can GCS)
model = None
if os.path.exists(MODEL_PATH):
    model = joblib.load(MODEL_PATH)
    print(f"Model da duoc tai tu {MODEL_PATH}")
else:
    print(f"Canh bao: Khong tim thay model tai {MODEL_PATH}. Chay train.py truoc.")


@app.get("/health")
def health():
    """
    Endpoint kiem tra suc khoe server.
    GitHub Actions goi endpoint nay sau khi deploy de xac nhan server dang chay.

    Tra ve: {"status": "ok"}
    """
    return {"status": "ok"}


@app.post("/predict")
def predict(req: PredictRequest):
    """
    Endpoint suy luan chinh.

    Dau vao : JSON {"features": [f1, f2, ..., f12]}
    Dau ra  : JSON {"prediction": <0|1|2>, "label": <"thap"|"trung_binh"|"cao">}

    Thu tu 12 dac trung (khop voi thu tu trong FEATURE_NAMES cua test):
        fixed_acidity, volatile_acidity, citric_acid, residual_sugar,
        chlorides, free_sulfur_dioxide, total_sulfur_dioxide, density,
        pH, sulphates, alcohol, wine_type
    """
    if model is None:
        raise HTTPException(status_code=503, detail="Model chua duoc tai. Vui long chay train.py truoc.")

    # TODO 6: Kiem tra so luong dac trung.
    if len(req.features) != 12:
        raise HTTPException(status_code=400, detail=f"Can dung 12 dac trung, nhan duoc {len(req.features)}.")

    # TODO 7: Goi model.predict([req.features]) de lay ket qua du doan.
    pred = model.predict([req.features])

    # TODO 8: Tra ve dict chua "prediction" (int) va "label" (string).
    label_map = {0: "thap", 1: "trung_binh", 2: "cao"}
    prediction = int(pred[0])
    return {"prediction": prediction, "label": label_map[prediction]}


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
