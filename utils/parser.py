import json

def get_top_preds(json_file_path):
    """
    Reads a JSON file and returns a simplified list where for each image,
    for each model, only the prediction with the highest confidence is included.
    
    Parameters:
        json_file_path (str): Path to the JSON file.

    Returns:
        list: A list of dictionaries for each image with highest-confidence predictions per model.
    
    Usage:
        from utils.parser import get_top_preds
        get_top_preds("<path-to-your-json>")
        
    """
    with open(json_file_path, 'r') as file:
        data = json.load(file)

    results = []

    for item in data:
        image_result = {
            "image": item["image"],
            "models": {}
        }

        for model_name, model_data in item.get("models", {}).items():
            predictions = model_data.get("predictions", [])
            if predictions:
                best_pred = max(predictions, key=lambda p: p["confidence"])
                image_result["models"][model_name] = best_pred

        results.append(image_result)

    return results
