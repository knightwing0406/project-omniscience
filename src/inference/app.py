import torch
import torch.nn.functional as F
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import List

# Import your custom modular pipeline blocks directly from your source package layers
from src.models.detector import EnterpriseAcousticClassifier
from src.data_pipeline.processor import AcousticSignalProcessor
from src.utils.logger import setup_production_logger

# 1. Initialize System Logging and Framework Components
logger = setup_production_logger("inference_server")
logger.info("⚡ [API INITIALIZATION] Booting up Project Omniscience Web Serving Gateway...")

app = FastAPI(
    title="Project Omniscience API Gateway",
    description="Enterprise REST API service exposing GPU-accelerated acoustic anomaly detection pipelines.",
    version="1.0.0"
)

# 2. In-Memory Component Vault Initialization
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = None
signal_processor = None

# Define the exact incoming structural data rules using a Pydantic schema validation model
class AudioPayload(BaseModel):
    raw_audio: List[float] # Strictly enforces that incoming packets must be lists of decimals

@app.on_event("startup")
def load_ecosystem_dependencies():
    """Executes background asset loading upon server boot, ensuring zero latency spikes on first call."""
    global model, signal_processor
    try:
        # Instantiate your signal transformation layers
        signal_processor = AcousticSignalProcessor()
        
        # Instantiate your neural network architecture structure
        model = EnterpriseAcousticClassifier().to(device)
        
        # Check if an optimized weights checkpoint artifact exists on disk
        checkpoint_path = "acoustic_model.pt"
        import os
        if os.path.exists(checkpoint_path):
            model.load_state_dict(torch.load(checkpoint_path, map_location=device))
            logger.info(f"💾 [API DEPLOYMENT] Successfully synced neural network weights checkpoint: [{checkpoint_path}]")
        else:
            logger.warning("⚠️ [API DEPLOYMENT] Post-training checkpoint not detected. Utilizing un-optimized model parameters.")
            
        model.eval() # Hard lock model parameters into inference mode (clips dropout branches)
        logger.info("🚀 [API READY] Web API Server dependencies compiled. Serving channels fully operational.")
    except Exception as e:
        logger.error(f"❌ [API FATAL CRASH] Core server boot failure: {str(e)}")
        raise e

@app.get("/")
def system_health_ping():
    """Provides an automated heartbeat validation check for network infrastructure controllers."""
    return {
        "status": "healthy",
        "compute_device": str(device),
        "platform": "Project Omniscience Acoustic Intelligence Server"
    }

@app.post("/predict")
def route_acoustic_stream_inference(payload: AudioPayload):
    """
    Receives raw 1D audio wave stream arrays, processes them into 2D maps via GPU,
    and returns a localized classification response vector detailing mechanical health.
    """
    if not payload.raw_audio:
        raise HTTPException(status_code=400, detail="Inbound payload contains an empty audio data vector.")
        
    try:
        # Step A: Transform incoming text array into a 2D Spectrogram Tensor image directly on the GPU
        spectrogram_image = signal_processor.process_stream_chunk(payload.raw_audio)
        
        # Step B: Pass spatial matrix forward through the Neural Network pathways
        with torch.no_grad():
            raw_logits = model(spectrogram_image)
            prediction_probabilities = F.softmax(raw_logits, dim=1)
            
        # Extract classification probability scores
        healthy_score = prediction_probabilities[0][0].item() * 100
        anomaly_score = prediction_probabilities[0][1].item() * 100
        final_prediction_state = torch.argmax(prediction_probabilities, dim=1).item()
        
        system_status_label = "CRITICAL_ANOMALY" if final_prediction_state == 1 else "NOMINAL_HEALTHY"
        target_confidence = anomaly_score if final_prediction_state == 1 else healthy_score
        
        # Dispatch metric trace log to rotating system history files
        logger.info(f"📊 [INFERENCE TRACE] Status resolved: {system_status_label} | Engine Confidence: {target_confidence:.2f}%")
        
        # Return standardized corporate JSON payload format back to client terminal
        return {
            "prediction": system_status_label,
            "confidence_score": round(target_confidence, 2),
            "telemetry_metrics": {
                "healthy_probability": round(healthy_score, 2),
                "anomaly_probability": round(anomaly_score, 2)
            }
        }
    except Exception as e:
        logger.error(f"🚨 [API RUNTIME EXCEPTION] Failed processing live data packet inference: {str(e)}")
        raise HTTPException(status_code=500, detail="Internal server anomaly encountered during deep learning execution.")
