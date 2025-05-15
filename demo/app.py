from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
import io
import base64
import logging
from PIL import Image
import uvicorn
import os

from src.digitrecognitionapp.utils.utils import (
    load_model,
    preprocess_image,
    predict_digit,
)

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Create FastAPI app
app = FastAPI(
    title="MNIST Digit Predictor API",
    description="A FastAPI backend for predicting handwritten digits using the MNIST dataset",
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

model_path = os.environ.get("MODEL_PATH", "dump/version_1/model.pth")
# Load the model
model = load_model(model_path)


@app.get("/")
async def root():
    """Root endpoint to check if API is running"""
    return {"message": "MNIST Digit Predictor API is running"}


@app.get("/health")
async def health_check():
    """Health check endpoint"""
    if model is not None:
        return {"status": "healthy", "model_loaded": True}
    return JSONResponse(
        status_code=503, content={"status": "unhealthy", "model_loaded": False}
    )


@app.post("/predict/image")
async def predict_from_image(file: UploadFile = File(...)):
    """
    Predict digit from uploaded image

    - **file**: An image file containing a handwritten digit

    Returns prediction results with confidence scores.
    """
    try:
        # Read the image file
        contents = await file.read()
        image = Image.open(io.BytesIO(contents))

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = predict_digit(model, processed_image)

        response = {"prediction": prediction}
        return response
    except Exception as e:
        logger.error(f"Error processing prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


@app.post("/predict/base64")
async def predict_from_base64(data: dict):
    """
    Predict digit from base64 encoded image

    - **data**: JSON object with 'image' field containing base64 encoded image

    Returns prediction results with confidence scores.
    """
    try:
        if "image" not in data:
            raise HTTPException(status_code=400, detail="No image data provided")

        # Decode base64 image
        base64_data = (
            data["image"].split(",")[1] if "," in data["image"] else data["image"]
        )
        image_data = base64.b64decode(base64_data)
        image = Image.open(io.BytesIO(image_data))

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Make prediction
        prediction = predict_digit(model, processed_image)

        response = {"prediction": prediction}
        return response
    except Exception as e:
        logger.error(f"Error processing base64 prediction: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction error: {str(e)}")


if __name__ == "__main__":
    # Run the API server
    port = int(os.environ.get("PORT", 8000))
    uvicorn.run("demo.app:app", host="0.0.0.0", port=port, reload=True)
