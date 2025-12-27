from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from model import predict
from logic import post_process

app = FastAPI(title="HF FastAPI Custom Prediction")

class PredictionRequest(BaseModel):
    text: str

class PredictionResponse(BaseModel):
    label: str
    confidence: float

@app.post("/predict", response_model=PredictionResponse)
def predict_text(req: PredictionRequest):
    # Input validation
    if len(req.text.strip()) < 5:
        raise HTTPException(
            status_code=400,
            detail="Text must be at least 5 characters"
        )

    raw_prediction = predict(req.text)
    final_prediction = post_process(raw_prediction)

    return final_prediction
