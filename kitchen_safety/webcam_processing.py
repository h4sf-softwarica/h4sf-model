import cv2
import time
from .model_manager import ModelManager  # Relative import
from .visualization import visualize_predictions  # Relative import
from .safety_advisor import SafetyAdvisor  # Relative import


def process_webcam(model="meat", confidence=0.3, enable_ai_analysis=True):
    """Real-time processing from webcam with AI insights"""
    print("\n Starting webcam processing...")
    model_manager = ModelManager()
    model_manager.current_model = model
    
    # Initialize safety advisor
    safety_advisor = SafetyAdvisor() if enable_ai_analysis else None
    ai_insight = ""
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print(" Could not access webcam")
        return
    
    print(" Webcam activated. Press 'q' to exit")
    print(" Press 'm' to cycle through models")
    print(" Press 'a' to trigger AI analysis")
    
    # For FPS calculation
    frame_times = deque(maxlen=10)
    last_time = time.time()
    last_analysis_time = 0
    ANALYSIS_INTERVAL = 10  # seconds
    
    frame_count = 0
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print(" Failed to capture frame")
            break
        
        frame_count += 1
        
        # Calculate FPS
        current_time = time.time()
        frame_time = current_time - last_time
        frame_times.append(frame_time)
        fps = 1 / (sum(frame_times) / len(frame_times)) if frame_times else 0
        last_time = current_time
        
        try:
            # Process frame
            start_inference = time.time()
            preds = model_manager.predict(frame, confidence=confidence)
            inference_time = time.time() - start_inference
            
            # Perform AI analysis periodically or on demand
            if safety_advisor:
                # Automatic periodic analysis
                if current_time - last_analysis_time > ANALYSIS_INTERVAL:
                    ai_insight = safety_advisor.get_proactive_alert(frame)
                    if ai_insight:
                        print(f"\n🔍 AI Insight: {ai_insight}")
                        last_analysis_time = current_time
            
            # Visualize predictions
            display_frame = visualize_predictions(
                frame, 
                preds, 
                model_manager.current_model,
                fps,
                inference_time,
                ai_insight
            )
            
            # Display frame
            cv2.imshow("Real-time Kitchen Safety Monitoring", display_frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):  # Quit
                break
            elif key == ord('m'):  # Cycle models
                models = list(model_manager.models.keys())
                current_idx = models.index(model_manager.current_model)
                next_idx = (current_idx + 1) % len(models)
                model_manager.current_model = models[next_idx]
                print(f" Switched to model: {model_manager.current_model}")
            elif key == ord('a') and safety_advisor:  # Trigger AI analysis
                ai_insight = safety_advisor.analyze_image(frame)
                print(f"\n🔍 Manual AI Analysis:\n{ai_insight}")
            elif key == ord('c'):  # Confidence down
                confidence = max(0.1, confidence - 0.05)
                print(f" Confidence threshold: {confidence:.2f}")
            elif key == ord('C'):  # Confidence up (shift+c)
                confidence = min(0.95, confidence + 0.05)
                print(f" Confidence threshold: {confidence:.2f}")
                
        except Exception as e:
            print(f" Error processing frame: {str(e)}")
    
    cap.release()
    cv2.destroyAllWindows()
    print(" Webcam processing stopped")