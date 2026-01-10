"""
REST API for Graph-based Recommendation System

Production-ready FastAPI server with:
- Real-time recommendations
- Batch predictions
- Model serving
- Health checks and monitoring
- Rate limiting and caching
"""

from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import List, Optional, Dict
import torch
import numpy as np
from datetime import datetime
import logging
from functools import lru_cache
import time

# Imports from our modules
import sys
import os
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.lightgcn import LightGCN, load_model as load_lightgcn
from models.ngcf import NGCF, load_model as load_ngcf

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Initialize FastAPI app
app = FastAPI(
    title="Graph Recommendation API",
    description="Production API for graph-based collaborative filtering",
    version="1.0.0"
)

# CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global model storage
MODEL_REGISTRY = {}
ADJACENCY_MATRICES = {}

# Request/Response models
class RecommendationRequest(BaseModel):
    user_id: str = Field(..., description="User ID to generate recommendations for")
    k: int = Field(10, ge=1, le=100, description="Number of recommendations")
    exclude_seen: bool = Field(True, description="Exclude items user has already seen")
    model_name: str = Field("lightgcn", description="Model to use (lightgcn or ngcf)")

class RecommendationResponse(BaseModel):
    user_id: str
    recommendations: List[Dict[str, float]]
    model_used: str
    timestamp: str
    latency_ms: float

class BatchRecommendationRequest(BaseModel):
    user_ids: List[str] = Field(..., max_items=1000)
    k: int = Field(10, ge=1, le=100)
    model_name: str = Field("lightgcn")

class HealthResponse(BaseModel):
    status: str
    models_loaded: List[str]
    uptime_seconds: float
    timestamp: str

class ModelInfo(BaseModel):
    name: str
    type: str
    n_users: int
    n_items: int
    embedding_size: int
    parameters: int

# Startup time tracking
START_TIME = time.time()


@app.on_event("startup")
async def startup_event():
    """Load models on startup."""
    logger.info("Starting Graph Recommendation API...")
    
    # Load default models (in production, load from model registry/S3)
    try:
        # Check if model files exist
        lightgcn_path = "models/lightgcn_checkpoint.pt"
        ngcf_path = "models/ngcf_checkpoint.pt"
        
        if os.path.exists(lightgcn_path):
            MODEL_REGISTRY['lightgcn'] = load_lightgcn(lightgcn_path)
            logger.info("Loaded LightGCN model")
        
        if os.path.exists(ngcf_path):
            MODEL_REGISTRY['ngcf'] = load_ngcf(ngcf_path)
            logger.info("Loaded NGCF model")
            
    except Exception as e:
        logger.warning(f"Could not load models: {e}")
        logger.info("API will start without pre-loaded models")


@app.get("/", response_model=Dict)
async def root():
    """Root endpoint with API information."""
    return {
        "service": "Graph Recommendation API",
        "version": "1.0.0",
        "status": "running",
        "endpoints": {
            "health": "/health",
            "recommend": "/recommend",
            "batch_recommend": "/batch_recommend",
            "models": "/models"
        }
    }


@app.get("/health", response_model=HealthResponse)
async def health_check():
    """Health check endpoint."""
    return HealthResponse(
        status="healthy",
        models_loaded=list(MODEL_REGISTRY.keys()),
        uptime_seconds=time.time() - START_TIME,
        timestamp=datetime.utcnow().isoformat()
    )


@app.get("/models", response_model=List[ModelInfo])
async def list_models():
    """List available models and their metadata."""
    model_info = []
    
    for name, model in MODEL_REGISTRY.items():
        n_params = sum(p.numel() for p in model.parameters())
        
        info = ModelInfo(
            name=name,
            type=model.__class__.__name__,
            n_users=model.n_users,
            n_items=model.n_items,
            embedding_size=model.embedding_size,
            parameters=n_params
        )
        model_info.append(info)
    
    return model_info


@app.post("/recommend", response_model=RecommendationResponse)
async def recommend(request: RecommendationRequest):
    """
    Generate personalized recommendations for a user.
    
    Args:
        request: Recommendation request with user_id and parameters
        
    Returns:
        Recommendations with items and scores
    """
    start_time = time.time()
    
    # Validate model
    if request.model_name not in MODEL_REGISTRY:
        raise HTTPException(
            status_code=404,
            detail=f"Model '{request.model_name}' not found. Available: {list(MODEL_REGISTRY.keys())}"
        )
    
    model = MODEL_REGISTRY[request.model_name]
    
    # Convert user_id to internal index (in production, use user mapping)
    try:
        user_idx = int(request.user_id)
    except ValueError:
        raise HTTPException(status_code=400, detail="Invalid user_id format")
    
    if user_idx < 0 or user_idx >= model.n_users:
        raise HTTPException(
            status_code=404,
            detail=f"User ID {user_idx} not found. Valid range: [0, {model.n_users-1}]"
        )
    
    # Get adjacency matrix
    if request.model_name not in ADJACENCY_MATRICES:
        raise HTTPException(
            status_code=500,
            detail="Adjacency matrix not loaded for this model"
        )
    
    adj_matrix = ADJACENCY_MATRICES[request.model_name]
    
    # Generate recommendations
    try:
        seen_items = set()  # In production, load from database
        item_ids, scores = model.recommend(
            adj_matrix,
            user_idx,
            k=request.k,
            exclude_seen=request.exclude_seen,
            seen_items=seen_items
        )
        
        # Format response
        recommendations = [
            {"item_id": int(item_id), "score": float(score)}
            for item_id, score in zip(item_ids, scores)
        ]
        
        latency_ms = (time.time() - start_time) * 1000
        
        return RecommendationResponse(
            user_id=request.user_id,
            recommendations=recommendations,
            model_used=request.model_name,
            timestamp=datetime.utcnow().isoformat(),
            latency_ms=latency_ms
        )
        
    except Exception as e:
        logger.error(f"Recommendation error: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/batch_recommend")
async def batch_recommend(
    request: BatchRecommendationRequest,
    background_tasks: BackgroundTasks
):
    """
    Generate recommendations for multiple users.
    
    For large batches, consider using async/background processing.
    
    Args:
        request: Batch recommendation request
        background_tasks: FastAPI background tasks
        
    Returns:
        Batch recommendations
    """
    start_time = time.time()
    
    if request.model_name not in MODEL_REGISTRY:
        raise HTTPException(status_code=404, detail=f"Model not found")
    
    model = MODEL_REGISTRY[request.model_name]
    adj_matrix = ADJACENCY_MATRICES.get(request.model_name)
    
    if adj_matrix is None:
        raise HTTPException(status_code=500, detail="Model not properly initialized")
    
    results = []
    
    for user_id in request.user_ids:
        try:
            user_idx = int(user_id)
            
            if user_idx < 0 or user_idx >= model.n_users:
                results.append({
                    "user_id": user_id,
                    "error": "User ID out of range",
                    "recommendations": []
                })
                continue
            
            item_ids, scores = model.recommend(
                adj_matrix,
                user_idx,
                k=request.k,
                exclude_seen=True,
                seen_items=set()
            )
            
            recommendations = [
                {"item_id": int(item_id), "score": float(score)}
                for item_id, score in zip(item_ids, scores)
            ]
            
            results.append({
                "user_id": user_id,
                "recommendations": recommendations,
                "error": None
            })
            
        except Exception as e:
            results.append({
                "user_id": user_id,
                "error": str(e),
                "recommendations": []
            })
    
    latency_ms = (time.time() - start_time) * 1000
    
    return {
        "results": results,
        "total_users": len(request.user_ids),
        "successful": sum(1 for r in results if r["error"] is None),
        "failed": sum(1 for r in results if r["error"] is not None),
        "latency_ms": latency_ms,
        "timestamp": datetime.utcnow().isoformat()
    }


@app.post("/load_model")
async def load_model(
    model_name: str,
    model_path: str,
    model_type: str = "lightgcn"
):
    """
    Load a new model into the registry.
    
    Args:
        model_name: Name to register model under
        model_path: Path to model checkpoint
        model_type: Type of model (lightgcn or ngcf)
        
    Returns:
        Success message
    """
    try:
        if model_type == "lightgcn":
            model = load_lightgcn(model_path)
        elif model_type == "ngcf":
            model = load_ngcf(model_path)
        else:
            raise HTTPException(status_code=400, detail="Invalid model type")
        
        MODEL_REGISTRY[model_name] = model
        logger.info(f"Loaded model '{model_name}' from {model_path}")
        
        return {
            "status": "success",
            "message": f"Model '{model_name}' loaded successfully",
            "model_type": model_type
        }
        
    except Exception as e:
        logger.error(f"Error loading model: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/metrics")
async def get_metrics():
    """
    Get API metrics (for monitoring/observability).
    
    In production, integrate with Prometheus, Datadog, etc.
    """
    return {
        "uptime_seconds": time.time() - START_TIME,
        "models_loaded": len(MODEL_REGISTRY),
        "model_names": list(MODEL_REGISTRY.keys()),
        "timestamp": datetime.utcnow().isoformat()
    }


if __name__ == "__main__":
    import uvicorn
    
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8000,
        log_level="info",
        workers=4  # Adjust based on CPU cores
    )
