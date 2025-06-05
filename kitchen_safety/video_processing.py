import cv2
import time
import os
import numpy as np
from collections import deque
from .model_manager import ModelManager  # Relative import
from .visualization import visualize_predictions  # Relative import
from .safety_advisor import SafetyAdvisor  # Relative import


def process_prerecorded_video(video_path, model="meat", confidence=0.3, 
                             save_output=False, frame_skip=15, output="output",
                             enable_ai_analysis=True):
    """Process a pre-recorded video file with Gemini AI analysis - simple and fast"""
    print(f"\nStarting video analysis: {video_path}")
    
    # Create output directory
    os.makedirs(output, exist_ok=True)
    
    # Initialize safety advisor for Gemini analysis
    safety_advisor = SafetyAdvisor() if enable_ai_analysis else None
    
    if not safety_advisor:
        print("AI analysis is disabled. Enable it to analyze videos.")
        return
    
    # Get video info for logging
    cap = cv2.VideoCapture(video_path)
    if cap.isOpened():
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        duration = total_frames / fps
        print(f"Video Info: {width}x{height}, {duration:.1f}s, {total_frames} frames")
        cap.release()
    
    start_time = time.time()
    
    # Send entire video to Gemini for analysis
    print("Sending video to Gemini AI for complete analysis...")
    video_analysis = safety_advisor.analyze_video(video_path)
    
    if video_analysis:
        print(f"\nAnalysis completed in {time.time() - start_time:.1f}s")
        
        # Save analysis to file
        analysis_path = os.path.join(output, "video_safety_analysis.txt")
        with open(analysis_path, 'w') as f:
            f.write("KITCHEN SAFETY VIDEO ANALYSIS\n")
            f.write("=" * 50 + "\n\n")
            f.write(f"Video: {os.path.basename(video_path)}\n")
            f.write(f"Analysis Date: {time.strftime('%Y-%m-%d %H:%M:%S')}\n")
            f.write(f"Processing Time: {time.time() - start_time:.1f}s\n\n")
            f.write("GEMINI AI SAFETY ANALYSIS:\n")
            f.write("-" * 30 + "\n\n")
            f.write(video_analysis)
        
        print(f"Analysis saved to: {analysis_path}")
        
        # Show summary
        print(f"\nVIDEO ANALYSIS COMPLETE")
        print(f"Total time: {time.time() - start_time:.1f}s")
        print(f"Results saved to: {output}")
        
        # Quick hazard detection summary
        hazard_keywords = ['danger', 'hazard', 'unsafe', 'risk', 'warning', 'caution', 'safety', 'concern']
        hazard_mentions = sum(1 for keyword in hazard_keywords 
                             if keyword.lower() in video_analysis.lower())
        
        if hazard_mentions > 0:
            print(f"Safety-related mentions detected: {hazard_mentions}")
        else:
            print("Analysis completed")
        
        return video_analysis
    
    else:
        print("Failed to analyze video with Gemini AI")
        return None