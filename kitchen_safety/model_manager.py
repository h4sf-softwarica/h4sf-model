from roboflow import Roboflow
from dotenv import load_dotenv
import os

load_dotenv()

# Define models to evaluate
MODELS = {
    "meat": ("meat-detection-86nhn", 2),
    "kitchen": ("kitchen-detection-ynls5", 2),
    "sausage": ("sausage_v2", 3)
}

class ModelManager:
    """Manages model loading and prediction"""
    def __init__(self):
        self.models = {}
        self.current_model = "meat"
        self.rf = Roboflow(api_key=os.getenv("ROBOFLOW_API_KEY"))
        self.load_all_models()
    
    def load_all_models(self):
        """Initialize all models"""
        print("\n Loading models...")
        for model_name, (project_name, version_num) in MODELS.items():
            try:
                project = self.rf.workspace("roboflow-universe").project(project_name)
                model = project.version(version_num).model
                self.models[model_name] = model
                print(f" Model loaded: {model_name} ({project_name} v{version_num})")
            except Exception as e:
                print(f" Failed to load model {model_name}: {str(e)}")
    
    def predict(self, frame, confidence=0.3):
        """Get predictions from current model"""
        if self.current_model not in self.models:
            return []
        
        try:
            result = self.models[self.current_model].predict(
                frame, 
                confidence=confidence,
                overlap=30
            ).json()
            return result.get('predictions', [])
        except Exception as e:
            print(f" Prediction error: {str(e)}")
            return []