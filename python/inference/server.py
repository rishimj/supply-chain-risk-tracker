from fastapi import FastAPI, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Dict, List, Optional, Any
import numpy as np
import pandas as pd
import logging
import time
import joblib
import os
import sys
from datetime import datetime
from pathlib import Path
import asyncio
import uvicorn

# Add the python directory to the path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.ensemble import SupplyChainRiskEnsemble
from features.feature_pipeline import FeaturePipeline

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(
    title="Supply Chain ML Inference Server",
    description="ML model serving for supply chain risk prediction",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Request/Response models
class PredictionRequest(BaseModel):
    company_id: str
    features: Dict[str, float]
    model_version: Optional[str] = "latest"

class BatchPredictionRequest(BaseModel):
    requests: List[PredictionRequest]
    model_version: Optional[str] = "latest"

class PredictionResponse(BaseModel):
    company_id: str
    risk_score: float = Field(..., ge=0, le=100, description="Risk score from 0-100")
    guidance_miss_probability: float = Field(..., ge=0, le=1, description="Probability of missing guidance")
    confidence: float = Field(..., ge=0, le=1, description="Model confidence")
    component_risks: Dict[str, float] = Field(..., description="Component-wise risk breakdown")
    feature_importance: Dict[str, float] = Field(..., description="Feature importance for this prediction")
    model_version: str
    prediction_id: str
    timestamp: datetime

class BatchPredictionResponse(BaseModel):
    predictions: List[PredictionResponse]
    batch_id: str
    total_predictions: int
    successful_predictions: int
    failed_predictions: int
    processing_time_ms: float

class ModelStatus(BaseModel):
    model_loaded: bool
    model_version: str
    model_path: str
    loaded_at: datetime
    predictions_served: int
    average_response_time_ms: float
    feature_pipeline_loaded: bool

class HealthResponse(BaseModel):
    status: str
    timestamp: datetime
    model_status: ModelStatus
    uptime_seconds: float

# Global variables for model and pipeline
model: Optional[SupplyChainRiskEnsemble] = None
feature_pipeline: Optional[FeaturePipeline] = None
model_metadata: Dict[str, Any] = {}
server_start_time = time.time()
prediction_count = 0
total_response_time = 0.0

async def load_model_and_pipeline():
    """Load the trained model and feature pipeline"""
    global model, feature_pipeline, model_metadata
    
    try:
        # Look for the latest model in the models directory
        models_dir = Path("models/artifacts")
        
        if not models_dir.exists():
            logger.warning("Models directory not found. Using mock model.")
            model = MockEnsembleModel()
            feature_pipeline = MockFeaturePipeline()
            model_metadata = {
                'model_version': 'mock_v1.0',
                'model_type': 'mock',
                'loaded_at': datetime.now().isoformat()
            }
            return
        
        # Find the latest model directory
        model_dirs = [d for d in models_dir.iterdir() if d.is_dir() and d.name.startswith('model_')]
        
        if not model_dirs:
            logger.warning("No trained models found. Using mock model.")
            model = MockEnsembleModel()
            feature_pipeline = MockFeaturePipeline()
            model_metadata = {
                'model_version': 'mock_v1.0',
                'model_type': 'mock',
                'loaded_at': datetime.now().isoformat()
            }
            return
        
        # Get the latest model (by directory name)
        latest_model_dir = sorted(model_dirs, key=lambda x: x.name)[-1]
        
        logger.info(f"Loading model from {latest_model_dir}")
        
        # Load model
        model_path = latest_model_dir / "ensemble_model.joblib"
        if model_path.exists():
            model = joblib.load(model_path)
            logger.info("Ensemble model loaded successfully")
        else:
            # Fallback to creating a new model instance
            model = SupplyChainRiskEnsemble({})
            logger.warning("Model file not found, created new instance")
        
        # Load feature pipeline
        pipeline_path = latest_model_dir / "feature_pipeline.joblib"
        if pipeline_path.exists():
            feature_pipeline = FeaturePipeline({})
            feature_pipeline.load_pipeline(str(pipeline_path))
            logger.info("Feature pipeline loaded successfully")
        else:
            feature_pipeline = FeaturePipeline({})
            logger.warning("Feature pipeline not found, created new instance")
        
        # Load metadata
        metadata_path = latest_model_dir / "metadata.json"
        if metadata_path.exists():
            import json
            with open(metadata_path, 'r') as f:
                model_metadata = json.load(f)
        else:
            model_metadata = {
                'model_version': latest_model_dir.name,
                'model_type': 'ensemble',
                'loaded_at': datetime.now().isoformat()
            }
        
        logger.info("Model and pipeline loaded successfully")
        
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        # Fallback to mock model
        model = MockEnsembleModel()
        feature_pipeline = MockFeaturePipeline()
        model_metadata = {
            'model_version': 'mock_v1.0',
            'model_type': 'mock_fallback',
            'loaded_at': datetime.now().isoformat(),
            'error': str(e)
        }

# Mock classes for fallback
class MockEnsembleModel:
    def __init__(self):
        self.model_version = "mock_v1.0"
        self.start_time = time.time()
        self.predictions_served = 0
        logger.info("MockEnsembleModel initialized")
    
    def predict(self, features_df: pd.DataFrame) -> Dict:
        """Generate mock prediction"""
        self.predictions_served += 1
        
        # Convert DataFrame to dict for processing
        if isinstance(features_df, pd.DataFrame):
            features = features_df.iloc[0].to_dict() if len(features_df) > 0 else {}
        else:
            features = features_df
        
        # Mock ensemble prediction logic
        financial_features = [v for k, v in features.items() if 'financial' in k.lower()]
        network_features = [v for k, v in features.items() if 'network' in k.lower() or 'supplier' in k.lower()]
        temporal_features = [v for k, v in features.items() if 'volatility' in k.lower() or 'trend' in k.lower()]
        sentiment_features = [v for k, v in features.items() if 'sentiment' in k.lower() or 'nlp' in k.lower()]
        
        # Calculate component risks
        financial_risk = np.mean(financial_features) if financial_features else 0.3
        network_risk = np.mean(network_features) if network_features else 0.4
        temporal_risk = np.mean(temporal_features) if temporal_features else 0.2
        sentiment_risk = np.mean(sentiment_features) if sentiment_features else 0.5
        
        # Normalize risks to 0-1 range
        financial_risk = max(0, min(1, financial_risk))
        network_risk = max(0, min(1, network_risk))
        temporal_risk = max(0, min(1, temporal_risk))
        sentiment_risk = max(0, min(1, sentiment_risk))
        
        # Calculate overall risk score
        risk_score = (
            financial_risk * 0.3 +
            network_risk * 0.25 +
            temporal_risk * 0.2 +
            sentiment_risk * 0.25
        ) * 100  # Convert to 0-100 scale
        
        # Convert to guidance miss probability
        guidance_miss_probability = 1 / (1 + np.exp(-5 * (risk_score/100 - 0.5)))
        
        # Calculate confidence (mock)
        confidence = 0.75 + np.random.random() * 0.2  # 0.75-0.95
        
        # Component risks
        component_risks = {
            "financial_risk": float(financial_risk * 100),
            "network_risk": float(network_risk * 100),
            "temporal_risk": float(temporal_risk * 100),
            "sentiment_risk": float(sentiment_risk * 100)
        }
        
        # Feature importance (mock)
        feature_importance = {}
        for feature in features.keys():
            feature_importance[feature] = np.random.random() * 0.1
        
        # Normalize feature importance
        total_importance = sum(feature_importance.values())
        if total_importance > 0:
            feature_importance = {k: v/total_importance for k, v in feature_importance.items()}
        
        return {
            'risk_score': float(risk_score),
            'guidance_miss_probability': float(guidance_miss_probability),
            'confidence': float(confidence),
            'component_risks': component_risks,
            'feature_importance': feature_importance,
            'model_version': self.model_version,
            'prediction_id': f"mock_{int(time.time())}_{np.random.randint(1000, 9999)}",
            'timestamp': datetime.now()
        }

class MockFeaturePipeline:
    def __init__(self):
        self.feature_names = None
        logger.info("MockFeaturePipeline initialized")
    
    def transform(self, features_df: pd.DataFrame) -> pd.DataFrame:
        """Mock feature transformation"""
        return features_df

@app.on_event("startup")
async def startup_event():
    """Initialize the model and pipeline on startup"""
    logger.info("Starting ML inference server...")
    await load_model_and_pipeline()
    logger.info("Server startup completed")

@app.post("/predict", response_model=PredictionResponse)
async def predict(request: PredictionRequest):
    """Make a single prediction"""
    global prediction_count, total_response_time
    
    start_time = time.time()
    
    try:
        if model is None:
            raise HTTPException(status_code=503, detail="Model not loaded")
        
        # Convert features to DataFrame
        features_df = pd.DataFrame([request.features])
        
        # Apply feature pipeline if available
        if feature_pipeline is not None:
            try:
                processed_features = feature_pipeline.transform(features_df)
            except Exception as e:
                logger.warning(f"Feature pipeline failed, using raw features: {e}")
                processed_features = features_df
        else:
            processed_features = features_df
        
        # Make prediction
        prediction = model.predict(processed_features)
        
        # Update statistics
        prediction_count += 1
        response_time = (time.time() - start_time) * 1000
        total_response_time += response_time
        
        return PredictionResponse(
            company_id=request.company_id,
            risk_score=prediction['risk_score'],
            guidance_miss_probability=prediction['guidance_miss_probability'],
            confidence=prediction['confidence'],
            component_risks=prediction['component_risks'],
            feature_importance=prediction['feature_importance'],
            model_version=prediction['model_version'],
            prediction_id=prediction['prediction_id'],
            timestamp=prediction['timestamp']
        )
    
    except Exception as e:
        logger.error(f"Prediction error: {e}")
        raise HTTPException(status_code=500, detail=f"Prediction failed: {str(e)}")

@app.post("/predict/batch", response_model=BatchPredictionResponse)
async def batch_predict(request: BatchPredictionRequest):
    """Make batch predictions"""
    start_time = time.time()
    batch_id = f"batch_{int(time.time())}_{np.random.randint(1000, 9999)}"
    
    predictions = []
    successful = 0
    failed = 0
    
    for pred_request in request.requests:
        try:
            # Make individual prediction
            prediction = await predict(pred_request)
            predictions.append(prediction)
            successful += 1
        except Exception as e:
            logger.error(f"Batch prediction failed for {pred_request.company_id}: {e}")
            failed += 1
    
    processing_time = (time.time() - start_time) * 1000
    
    return BatchPredictionResponse(
        predictions=predictions,
        batch_id=batch_id,
        total_predictions=len(request.requests),
        successful_predictions=successful,
        failed_predictions=failed,
        processing_time_ms=processing_time
    )

@app.get("/health", response_model=HealthResponse)
async def health():
    """Health check endpoint"""
    global prediction_count, total_response_time, server_start_time
    
    avg_response_time = total_response_time / prediction_count if prediction_count > 0 else 0.0
    uptime = time.time() - server_start_time
    
    model_status = ModelStatus(
        model_loaded=model is not None,
        model_version=model_metadata.get('model_version', 'unknown'),
        model_path=model_metadata.get('model_path', 'unknown'),
        loaded_at=datetime.fromisoformat(model_metadata.get('loaded_at', datetime.now().isoformat())),
        predictions_served=prediction_count,
        average_response_time_ms=avg_response_time,
        feature_pipeline_loaded=feature_pipeline is not None
    )
    
    status = "healthy" if model is not None else "degraded"
    
    return HealthResponse(
        status=status,
        timestamp=datetime.now(),
        model_status=model_status,
        uptime_seconds=uptime
    )

@app.get("/model/status")
async def get_model_status():
    """Get detailed model status"""
    return {
        "model_loaded": model is not None,
        "feature_pipeline_loaded": feature_pipeline is not None,
        "metadata": model_metadata,
        "predictions_served": prediction_count,
        "average_response_time_ms": total_response_time / prediction_count if prediction_count > 0 else 0.0
    }

@app.post("/model/reload")
async def reload_model():
    """Reload the model and feature pipeline"""
    try:
        await load_model_and_pipeline()
        return {"status": "success", "message": "Model reloaded successfully"}
    except Exception as e:
        logger.error(f"Model reload failed: {e}")
        raise HTTPException(status_code=500, detail=f"Model reload failed: {str(e)}")

@app.get("/model/metrics")
async def get_model_metrics():
    """Get model performance metrics"""
    if 'test_metrics' in model_metadata:
        return model_metadata['test_metrics']
    else:
        return {"message": "No metrics available"}

@app.get("/features/importance")
async def get_feature_importance():
    """Get feature importance from the model"""
    try:
        if hasattr(model, 'get_feature_importance'):
            importance = model.get_feature_importance()
            return importance
        elif feature_pipeline and hasattr(feature_pipeline, 'get_feature_importance'):
            importance = feature_pipeline.get_feature_importance()
            return importance
        else:
            return {"message": "Feature importance not available"}
    except Exception as e:
        logger.error(f"Failed to get feature importance: {e}")
        raise HTTPException(status_code=500, detail=str(e))

if __name__ == "__main__":
    uvicorn.run(
        "server:app",
        host="0.0.0.0",
        port=8001,
        reload=False,
        log_level="info"
    ) 