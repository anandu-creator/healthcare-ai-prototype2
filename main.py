# app/main.py
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Dict, List, Optional
import uvicorn

from app.models.diagnostic_model import DiagnosticModel
from app.models.clinical_agent import ClinicalAgent
from app.data.processors import MedicalDataProcessor


app = FastAPI(title="Healthcare AI Diagnostic API", version="0.1.0")

# Enable CORS for frontend integration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global models
diagnostic_model = DiagnosticModel()
clinical_agent = None
data_processor = MedicalDataProcessor()

class PatientData(BaseModel):
    patient_id: Optional[str] = None
    age: float
    gender: str
    temperature: float
    heart_rate: float
    bp_systolic: float
    bp_diastolic: float
    chest_pain: int
    shortness_breath: int
    fatigue: int

class DiagnosisResponse(BaseModel):
    patient_id: str
    timestamp: str
    diagnostic_predictions: List
    clinical_assessment: Dict
    recommendations: Dict
    confidence_score: float
    explanation: str

@app.on_event("startup")
async def startup_event():
    """Initialize models on startup"""
    global clinical_agent
    
    # Create and train model with sample data
    print("Initializing AI models...")
    sample_data = data_processor.create_sample_data(1000)
    X, y, feature_names = data_processor.preprocess_patient_data(sample_data)
    
    diagnostic_model.train(X, y, feature_names)
    clinical_agent = ClinicalAgent(diagnostic_model)
    
    print("Models initialized successfully!")

@app.get("/")
async def root():
    return {"message": "Healthcare AI Diagnostic API", "status": "running"}

@app.post("/diagnose", response_model=DiagnosisResponse)
async def diagnose_patient(patient_data: PatientData):
    """Analyze patient and provide diagnostic recommendations"""
    if not clinical_agent:
        raise HTTPException(status_code=500, detail="Models not initialized")
    
    try:
        # Convert to dict
        patient_dict = patient_data.dict()
        
        # Analyze patient
        result = clinical_agent.analyze_patient(patient_dict)
        
        return DiagnosisResponse(**result)
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/model/performance")
async def get_model_performance():
    """Get model performance metrics"""
    if not diagnostic_model.is_trained:
        raise HTTPException(status_code=404, detail="Model not trained")
    
    feature_importance = diagnostic_model.get_feature_importance()
    
    return {
        "model_type": "Random Forest Classifier",
        "feature_importance": feature_importance,
        "classes": list(diagnostic_model.class_names)
    }

@app.post("/retrain")
async def retrain_model():
    """Retrain model with new data"""
    try:
        # Generate new training data
        sample_data = data_processor.create_sample_data(1500)
        X, y, feature_names = data_processor.preprocess_patient_data(sample_data)
        
        # Retrain model
        diagnostic_model.train(X, y, feature_names)
        
        return {"message": "Model retrained successfully"}
    
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Retraining failed: {str(e)}")

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)