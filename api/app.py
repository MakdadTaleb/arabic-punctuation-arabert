import torch
import logging
import os
from contextlib import asynccontextmanager

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from src.inference.predictor import PunctuationPredictor  
import yaml


# ======================
# Logging
# ======================
logger = logging.getLogger("punctuation-api")
logging.basicConfig(level=logging.INFO)


# ======================
# Config Loader
# ======================
def load_config(path="config.yaml"):
    with open(path, "r") as f:
        return yaml.safe_load(f)


# ======================
# Lifespan
# ======================
@asynccontextmanager
async def lifespan(app: FastAPI):

    try:
        logger.info("Starting Punctuation API...")

        config = load_config(os.getenv("CONFIG_PATH", "config.yaml"))

        MODEL_PATH = config["model"]["save_path"]

        if not os.path.exists(MODEL_PATH):
            logger.warning(f"Model weights not found at {MODEL_PATH}. API in degraded mode.")
            app.state.predictor = None
            yield
            return

        predictor = PunctuationPredictor(
            model_path=MODEL_PATH,
            config_path=os.getenv("CONFIG_PATH", "config.yaml")
        )

        app.state.predictor = predictor

        logger.info(f"Punctuation API ready. Model: {predictor.model_name}")
        yield

    except Exception as e:
        logger.exception("Startup failed")
        raise e


# ======================
# FastAPI App
# ======================
app = FastAPI(
    title="Arabic Punctuation Restoration API",
    version="1.0.0",
    lifespan=lifespan,
)


# ======================
# Schemas
# ======================
class PredictRequest(BaseModel):
    text: str


class PredictResponse(BaseModel):
    original_text: str
    punctuated_text: str


# ======================
# Health Endpoint
# ======================
@app.get("/health")
def health():
    return {
        "status": "ok",
        "model_loaded": app.state.predictor is not None
    }


# ======================
# Predict Endpoint
# ======================
@app.post("/predict", response_model=PredictResponse)
def predict(request: PredictRequest):

    if app.state.predictor is None:
        raise HTTPException(
            status_code=503,
            detail="Model not initialized"
        )

    try:
        output_text = app.state.predictor.predict(request.text)
    except Exception:
        logger.exception("Inference error")
        raise HTTPException(status_code=500, detail="Inference failed")

    return PredictResponse(
        original_text=request.text,
        punctuated_text=output_text
    )