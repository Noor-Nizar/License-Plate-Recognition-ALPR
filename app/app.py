from typing import List, Optional, Dict
from fastapi import FastAPI
from fastapi.responses import JSONResponse
from .model import (
    predict_single_sample, 
    get_detection_model, 
    get_ocr, 
    ocr_easy_interface, 
    ocr_llm_interface,
    character_error_rate,
)

from .logger_config import logger
import torch

app = FastAPI()

# --- Configuration ---
device = 'cuda' if torch.cuda.is_available() else 'cpu'

# Load models only once on startup
detection_model = get_detection_model()
processor, ocr_model = get_ocr()
ocr_interface = ocr_easy_interface()  # or ocr_llm_interface(processor, ocr_model, device)
ocr_model = ocr_model.to(device) 

# --- Pydantic Schemas for Request/Response Validation ---
from pydantic import BaseModel, Field

class PredictionRequest(BaseModel):
    """Request body for the /predict endpoint."""
    images: List[str] = Field(..., description="List of image paths")
    ground_truth: Optional[List[str]] = Field(None, description="Optional list of ground truth texts (in the same order as images)")

class PredictionResponse(BaseModel):
    """Response body for the /predict endpoint."""
    predictions: Dict[int, str] = Field(..., description="Dictionary of predicted texts (in the same order as input images)")
    cer: Optional[float] = Field(None, description="Average Character Error Rate (if ground truth is provided)")
    # id: str = Field(..., description="ID of the prediction")

# --- API Endpoints ---

@app.post("/predict", response_model=PredictionResponse)
def predict_endpoint(request: PredictionRequest):
    """
    Endpoint for OCR prediction.

    Accepts a list of image paths and (optionally) ground truth texts.
    Processes images one by one for OCR.
    Returns predicted texts and character error rate (if ground truth is provided).
    """

    img_paths = request.images 
    predictions = predict_single_sample(img_paths, detection_model, ocr_interface, device=device)
    logger.info(predictions)
    cer = None
    if request.ground_truth:
        # Calculate CER if ground truth is provided
        cer = character_error_rate(request.ground_truth, [prediction for prediction in predictions.values()]) 

    return JSONResponse(content={"predictions": predictions, "cer": cer})

# --- Optional Health Check ---
@app.get("/health")
def health_check():
    return {"status": "ok"}