"""
FastAPI routes with API Key Authentication.
"""

import os
import logging
import tempfile
from pathlib import Path
from typing import List, Optional, Dict, Any

from fastapi import FastAPI, HTTPException, Response, File, UploadFile, Security, Depends
from fastapi.security import APIKeyHeader
from pydantic import BaseModel, validator
import numpy as np
import time

from src.model.inference_2d import HSI2DClassifier

logger = logging.getLogger(__name__)

# ---------- API Keys ----------
API_KEYS = {
    os.getenv("API_KEY_1", "test-key-1"): "client_1",
    os.getenv("API_KEY_2", "test-key-2"): "client_2",
}

api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

def validate_api_key(api_key: str = Security(api_key_header)):
    if api_key is None:
        raise HTTPException(status_code=403, detail="Missing API Key")
    if api_key not in API_KEYS:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return API_KEYS[api_key]

classifier = None

# ---------- Pydantic Models ----------
class HSIInput(BaseModel):
    image_data: List[List[List[float]]]

    @validator('image_data')
    def validate_image(cls, v):
        if not v:
            raise ValueError("Image data cannot be empty")
        if len(v) != 15:
            raise ValueError(f"Expected height 15, got {len(v)}")
        if len(v[0]) != 15:
            raise ValueError(f"Expected width 15, got {len(v[0])}")
        if len(v[0][0]) != 155:
            raise ValueError(f"Expected 155 bands, got {len(v[0][0])}")
        return v

class HSIInputBatch(BaseModel):
    images_data: List[List[List[List[float]]]]

    @validator('images_data')
    def validate_batch(cls, v):
        if not v:
            raise ValueError("Batch cannot be empty")
        if len(v) > 100:
            raise ValueError("Maximum batch size is 100")
        for i, img in enumerate(v):
            if len(img) != 15:
                raise ValueError(f"Image {i}: height must be 15")
            if len(img[0]) != 15:
                raise ValueError(f"Image {i}: width must be 15")
            if len(img[0][0]) != 155:
                raise ValueError(f"Image {i}: bands must be 155")
        return v

# ---------- FastAPI App ----------
def create_app(model_path: str) -> FastAPI:
    global classifier

    app = FastAPI(
        title="HSI 2D CNN Classifier API",
        description="Secure API for nutrient stress classification",
        version="3.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )

    @app.on_event("startup")
    async def startup_event():
        global classifier
        try:
            classifier = HSI2DClassifier(model_path)
            logger.info(f"Model loaded from {model_path}")
        except Exception as e:
            logger.error(f"Failed to load model: {e}")
            raise

    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "model_loaded": classifier is not None}

    @app.get("/model/info", dependencies=[Depends(validate_api_key)])
    async def get_model_info():
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        return {
            "model_type": "2D CNN Multiclassifier",
            "input_shape": classifier.input_shape,
            "num_classes": classifier.num_classes,
            "classes": classifier.class_names
        }

    @app.post("/predict", dependencies=[Depends(validate_api_key)])
    async def predict(input_data: HSIInput):
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            image = np.array(input_data.image_data, dtype=np.float32)
            return classifier.predict(image)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/predict/batch", dependencies=[Depends(validate_api_key)])
    async def predict_batch(input_data: HSIInputBatch):
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            images = np.array(input_data.images_data, dtype=np.float32)
            return classifier.predict_batch(images)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    @app.post("/predict/file", dependencies=[Depends(validate_api_key)])
    async def predict_from_file(file: UploadFile = File(...)):
        if classifier is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        try:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(file.filename)[1]) as tmp:
                content = await file.read()
                tmp.write(content)
                tmp_path = tmp.name

            result = classifier.predict_from_file(tmp_path)
            os.unlink(tmp_path)
            return result
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))

    return app