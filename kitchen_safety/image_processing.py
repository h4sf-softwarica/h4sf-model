import os
import glob
import time
import json
import cv2
from .model_manager import ModelManager
from .visualization import visualize_predictions
from .safety_advisor import SafetyAdvisor
from .hazard_detector import save_hazard_report
from .parser import save_simplified_results
from .config import DETECTION_THRESHOLD

def process_test_images(image_folder, model="meat", confidence=0.3, output="output", 
                       enable_ai_analysis=True, hazard_report=True,
                       gemini_fallback=True):
    """Process test images with Gemini fallback detection"""
    print("\n Starting test image processing...")
    model_manager = ModelManager()
    model_manager.current_model = model
    
    # Initialize safety advisor
    safety_advisor = SafetyAdvisor() if enable_ai_analysis else None
    
    # Get all test images
    image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.bmp']
    all_image_paths = []
    for ext in image_extensions:
        all_image_paths.extend(glob.glob(os.path.join(image_folder, ext), recursive=True))
    
    if not all_image_paths:
        print(" No images found in the specified folder!")
        return
    
    print(f" Found {len(all_image_paths)} test images")
    
    all_results = []
    
    for img_path in all_image_paths:
        print(f"\nProcessing: {os.path.basename(img_path)}")
        image = cv2.imread(img_path)
        if image is None:
            print(f" Couldn't load image: {img_path}")
            continue
        
        image_result = {
            "image": os.path.basename(img_path),
            "models": {}
        }
        
        # Track all detections for Gemini fallback
        all_detections = []
        
        for model_name in model_manager.models.keys():
            model_manager.current_model = model_name
            try:
                start_time = time.time()
                
                # Get prediction
                preds = model_manager.predict(image, confidence=confidence)
                inference_time = time.time() - start_time
                
                # Store results
                model_detections = []
                for pred in preds:
                    detection = {
                        "class": pred["class"],
                        "confidence": round(pred["confidence"], 3),
                        "bbox": {
                            "x": pred["x"],
                            "y": pred["y"],
                            "w": pred["width"],
                            "h": pred["height"]
                        }
                    }
                    model_detections.append(detection)
                    all_detections.append(detection)
                
                image_result["models"][model_name] = {
                    "predictions": model_detections,
                    "prediction_count": len(preds),
                    "inference_time": round(inference_time, 3)
                }
                
                print(f"  {model_name}: {len(preds)} detections in {inference_time:.2f}s")
                
            except Exception as e:
                print(f"     Error in {model_name}: {str(e)}")
                image_result["models"][model_name] = {
                    "error": str(e)
                }
        
        # Gemini fallback for undetected objects
        if safety_advisor and gemini_fallback and all_detections:
            try:
                # Get existing detected classes
                existing_classes = {d["class"].lower() for d in all_detections}
                
                # Use Gemini to detect additional safety objects
                gemini_objects = safety_advisor.detect_objects(image, all_detections)
                
                if gemini_objects:
                    print(f"  Gemini: Detected {len(gemini_objects)} additional objects")
                    
                    # Create Gemini model entry
                    gemini_detections = []
                    for obj in gemini_objects:
                        gemini_detections.append({
                            "class": obj,
                            "confidence": 1.0,  # Gemini doesn't provide confidence
                            "from_gemini": True
                        })
                    
                    image_result["models"]["gemini"] = {
                        "predictions": gemini_detections,
                        "prediction_count": len(gemini_objects),
                        "inference_time": 0
                    }
            except Exception as e:
                print(f"     Gemini fallback failed: {str(e)}")
        
        # Perform AI analysis if enabled
        if safety_advisor:
            try:
                analysis = safety_advisor.analyze_image(image)
                print(f" AI Analysis:\n{analysis}")
                image_result["ai_analysis"] = analysis
            except Exception as e:
                print(f" AI analysis error: {str(e)}")
        
        all_results.append(image_result)
    
    # Save full results
    os.makedirs(output, exist_ok=True)
    output_path = os.path.join(output, "test_results.json")
    with open(output_path, 'w') as f:
        json.dump(all_results, f, indent=4)
    
    # Generate hazard report if requested
    if hazard_report:
        hazard_dir = os.path.join(output, "hazard_reports")
        save_hazard_report(output_path, hazard_dir)
    
    # Generate and save simplified results
    simplified_path = save_simplified_results(output_path, output)
    
    # Save AI insights if enabled
    if safety_advisor:
        insights_path = os.path.join(output, "safety_insights.txt")
        with open(insights_path, 'w') as f:
            for result in all_results:
                if "ai_analysis" in result:
                    f.write(f"Image: {result['image']}\n")
                    f.write(f"Analysis: {result['ai_analysis']}\n")
                    f.write("-" * 50 + "\n")
    
    print(f"\n Test complete!")
    print(f"- Full results: {output_path}")
    print(f"- Simplified results: {simplified_path}")
    if hazard_report:
        print(f"- Hazard reports: {hazard_dir}")
    if safety_advisor:
        print(f"- Safety insights: {insights_path}")