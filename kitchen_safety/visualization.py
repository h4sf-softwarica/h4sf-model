import cv2

def visualize_predictions(frame, predictions, model_name, fps=0, inference_time=0, ai_insight=""):
    """Draw predictions on the frame with bounding boxes and labels"""
    # Create a copy to avoid modifying original
    display_frame = frame.copy()
    
    # Draw predictions
    for pred in predictions:
        try:
            # Convert center-based to corner-based coordinates
            x = int(pred["x"] - pred["width"] / 2)
            y = int(pred["y"] - pred["height"] / 2)
            w = int(pred["width"])
            h = int(pred["height"])
            
            # Draw bounding box
            color = (0, 255, 0)  # Green
            thickness = 2
            cv2.rectangle(display_frame, (x, y), (x + w, y + h), color, thickness)
            
            # Create label background
            label = f"{pred['class']} {pred['confidence']:.2f}"
            (text_width, text_height), _ = cv2.getTextSize(
                label, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1
            )
            cv2.rectangle(
                display_frame, 
                (x, y - text_height - 10), 
                (x + text_width, y), 
                color, 
                -1
            )
            
            # Put text on background
            cv2.putText(
                display_frame, 
                label, 
                (x, y - 5), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.5, 
                (0, 0, 0), 
                1
            )
        except KeyError as e:
            print(f" Missing key in prediction: {e}")
    
    # Display model info and performance
    info_line1 = f"Model: {model_name}"
    info_line2 = f"FPS: {fps:.1f} | Inference: {inference_time*1000:.1f}ms"
    
    cv2.putText(
        display_frame, 
        info_line1, 
        (10, 30), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 0, 255), 
        2
    )
    
    cv2.putText(
        display_frame, 
        info_line2, 
        (10, 60), 
        cv2.FONT_HERSHEY_SIMPLEX, 
        0.7, 
        (0, 255, 255), 
        2
    )
    
    # Display AI insight if available
    if ai_insight:
        # Wrap text to fit screen
        max_width = display_frame.shape[1] - 20
        font_size = 0.5
        thickness = 1
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Split insight into multiple lines
        words = ai_insight.split()
        lines = []
        current_line = ""
        
        for word in words:
            test_line = current_line + word + " "
            (w, _), _ = cv2.getTextSize(test_line, font, font_size, thickness)
            if w < max_width:
                current_line = test_line
            else:
                lines.append(current_line)
                current_line = word + " "
        lines.append(current_line)
        
        # Display each line
        y_pos = display_frame.shape[0] - 30 - (len(lines) * 25)
        for i, line in enumerate(lines):
            cv2.putText(
                display_frame, 
                line, 
                (10, y_pos + i*25), 
                font, 
                font_size, 
                (0, 255, 255), 
                thickness
            )
    
    return display_frame