import json
import os

def get_top_preds(data):
    """
    Processes detection data and returns simplified results with only the highest 
    confidence prediction per model per image.
    
    Parameters:
        data (list): List of detection results

    Returns:
        list: Simplified results
    """
    results = []

    for item in data:
        image_result = {
            "image": item["image"],
            "models": {}
        }

        for model_name, model_data in item.get("models", {}).items():
            predictions = model_data.get("predictions", [])
            if predictions:
                # Find highest confidence detection
                best_pred = max(
                    (p for p in predictions if "confidence" in p), 
                    key=lambda p: p["confidence"],
                    default=None
                )
                
                if best_pred:
                    image_result["models"][model_name] = {
                        "class": best_pred["class"],
                        "confidence": best_pred["confidence"],
                        "from_gemini": best_pred.get("from_gemini", False)
                    }
            else:
                image_result["models"][model_name] = "No detections"

        results.append(image_result)

    return results

def save_simplified_results(input_path, output_dir):
    """
    Processes a JSON results file and saves simplified results to output directory
    
    Parameters:
        input_path (str): Path to input JSON file
        output_dir (str): Directory to save simplified results
    """
    # Create output directory if needed
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate output filename
    filename = os.path.basename(input_path)
    simplified_path = os.path.join(output_dir, f"simplified_{filename}")
    
    # Load and process data
    with open(input_path, 'r') as f:
        data = json.load(f)
    
    simplified = get_top_preds(data)
    
    # Save simplified results
    with open(simplified_path, 'w') as f:
        json.dump(simplified, f, indent=4)
    
    print(f"Simplified results saved to: {simplified_path}")
    return simplified_path