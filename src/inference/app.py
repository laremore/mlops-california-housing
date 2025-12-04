
from fastapi import FastAPI, HTTPException, Request
from fastapi.responses import JSONResponse
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import uuid
import logging
from typing import Dict, Any

from .schemas import (
    HealthResponse, PredictionRequest, PredictionResponse,
    BatchPredictionRequest, BatchPredictionResponse
)
from .model_loader import ModelLoader
from .config import settings


logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title=settings.APP_NAME,
    version=settings.VERSION,
    description="ML service for predicting California housing prices",
    docs_url="/docs",
    redoc_url="/redoc"
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


model_loader = ModelLoader(str(settings.MODEL_PATH))

@app.on_event("startup")
async def startup_event():
    """Загружаем модель при старте сервиса."""
    logger.info("Starting up the service...")
    
    import os
    model_path = str(settings.MODEL_PATH)
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model exists: {os.path.exists(model_path)}")
    
    success = model_loader.load()
    if success:
        logger.info("Model loaded successfully")
        logger.info(f"Model info: {model_loader.get_info()}")
    else:
        logger.error("FAILED to load model!")
@app.get("/", include_in_schema=False)
async def root():
    
    return {
        "message": "California Housing Price Prediction API",
        "version": settings.VERSION,
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    model_info = model_loader.get_info()
    
    model_version = model_info.get("model_version", "1.0")
    model_loaded = model_info.get("is_loaded", False)
    
    if model_loaded:
        status = "healthy"
    else:
        status = "unhealthy (model not loaded)"
    
    return HealthResponse(
        status=status,
        model_loaded=model_loaded,
        model_version=model_version
    )

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        features = request.dict()
        
        prediction = model_loader.predict(features)
        
        request_id = str(uuid.uuid4())[:8]
        
        return PredictionResponse(
            prediction=prediction,
            model_version=model_loader.model_version,  # <-- Это должно работать
            request_id=request_id
        )
        
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def predict_batch(request: BatchPredictionRequest):

    if not model_loader.is_loaded:
        raise HTTPException(status_code=503, detail="Model is not loaded")
    
    try:
        features_list = [item.dict() for item in request.instances]
        
        if not features_list:
            raise HTTPException(status_code=400, detail="Empty batch")
        
        predictions = model_loader.predict_batch(features_list)
        
        return BatchPredictionResponse(
            predictions=predictions,
            model_version=model_loader.model_version,
            total_processed=len(predictions)
        )
        
    except Exception as e:
        logger.error(f"Batch prediction error: {e}")
        raise HTTPException(status_code=400, detail=str(e))


@app.exception_handler(Exception)
async def generic_exception_handler(request: Request, exc: Exception):
    logger.error(f"Unhandled exception: {exc}")
    return JSONResponse(
        status_code=500,
        content={"detail": "Internal server error"}
    )

if __name__ == "__main__":
    uvicorn.run(
        "app:app",
        host=settings.HOST,
        port=settings.PORT,
        reload=True
    )

@app.get("/debug-prediction")
async def debug_prediction():
    """Endpoint для отладки предсказания."""
    from .model_loader import model_loader
    
    test_features = {
        "longitude": -122.23,
        "latitude": 37.88,
        "housing_median_age": 41.0,
        "total_rooms": 880.0,
        "total_bedrooms": 129.0,
        "population": 322.0,
        "households": 126.0,
        "median_income": 8.3252,
        "bedrooms_per_room": 0.1466,
        "population_per_household": 2.5556,
        "households_per_population": 0.3913
    }
    
    import logging
    logger = logging.getLogger(__name__)
    
    logger.info("=== DEBUG PREDICTION ===")
    logger.info(f"Model loaded: {model_loader.is_loaded}")
    
    if model_loader.is_loaded:
        df = pd.DataFrame([test_features])
        logger.info(f"DataFrame shape: {df.shape}")
        logger.info(f"Columns: {df.columns.tolist()}")
        
        raw_pred = model_loader.model.predict(df)[0]
        logger.info(f"Raw prediction from model: {raw_pred}")
        
        price_from_exp = np.exp(raw_pred) - 1
        logger.info(f"After exp-1: ${price_from_exp:,.2f}")
        
        final_price = max(50000, min(price_from_exp, 500000))
        logger.info(f"After min/max: ${final_price:,.2f}")
        
        normal_pred = model_loader.predict(test_features)
        logger.info(f"Normal predict result: ${normal_pred:,.2f}")
        
        return {
            "raw_prediction": float(raw_pred),
            "price_from_exp": float(price_from_exp),
            "final_price": float(final_price),
            "normal_predict": float(normal_pred),
            "model_type": type(model_loader.model).__name__,
            "features_used": list(test_features.keys())
        }
    
    return {"error": "Model not loaded"}