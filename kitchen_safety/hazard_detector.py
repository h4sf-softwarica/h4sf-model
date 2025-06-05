import json
import os

# Define critical safety hazards we want to detect
SAFETY_HAZARDS = {
    "knife", "fire", "flame", "raw_meat", "raw_chicken", "raw_fish",
    "chemical", "spill", "slippery", "contamination", "bacteria",
    "unrefrigerated", "unsanitary", "mold", "pest", "rodent", "insect"
}

def is_hazard_detected(prediction):
    """
    Check if a prediction contains safety hazard keywords.
    
    Parameters:
        prediction (dict): Prediction dictionary with 'class' and 'confidence'
        
    Returns:
        bool: True if hazard detected, False otherwise
    """
    # Normalize class name for case-insensitive matching
    class_name = prediction["class"].lower()
    
    # Check against known hazards
    return any(hazard in class_name for hazard in SAFETY_HAZARDS)

def get_hazard_predictions(data):
    """
    Processes detection data and returns only images with detected safety hazards,
    including only the highest confidence hazard per model.
    
    Parameters:
        data (list): List of detection results

    Returns:
        list: Hazard report data
    """
    results = []

    for item in data:
        # Skip images without models data
        if "models" not in item:
            continue
            
        has_hazard = False
        image_result = {
            "image": item["image"],
            "hazards": {}
        }

        for model_name, model_data in item["models"].items():
            # Skip models without predictions
            if "predictions" not in model_data:
                continue
                
            predictions = model_data["predictions"]
            if not predictions:
                continue
                
            # Find the top hazard prediction for this model
            top_hazard = None
            for pred in predictions:
                if is_hazard_detected(pred):
                    if top_hazard is None or pred["confidence"] > top_hazard["confidence"]:
                        top_hazard = pred
            
            # If we found a hazard, add it to results
            if top_hazard:
                has_hazard = True
                image_result["hazards"][model_name] = {
                    "class": top_hazard["class"],
                    "confidence": top_hazard["confidence"],
                    "bbox": {
                        "x": top_hazard.get("x", 0),
                        "y": top_hazard.get("y", 0),
                        "w": top_hazard.get("width", top_hazard.get("w", 0)),
                        "h": top_hazard.get("height", top_hazard.get("h", 0))
                    }
                }
        
        # Only include images with at least one hazard
        if has_hazard:
            results.append(image_result)

    return results

def save_hazard_report(input_path, output_dir):
    """
    Processes a JSON results file and saves hazard report to output directory
    
    Parameters:
        input_path (str): Path to input JSON file
        output_dir (str): Directory to save hazard report
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    filename = os.path.basename(input_path)
    report_path = os.path.join(output_dir, f"hazard_report_{filename}")
    
    # Load and process data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    hazards = get_hazard_predictions(data)
    
    # Save hazard report
    with open(report_path, 'w') as f:
        json.dump(hazards, f, indent=4)
    
    print(f"Hazard report saved to: {report_path}")
    print(f"Detected {len(hazards)} images with safety hazards")
    return report_path