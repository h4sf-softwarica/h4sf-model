import google.generativeai as genai
from dotenv import load_dotenv
import os
import cv2
import numpy as np
import time
from PIL import Image
from .config import SAFETY_HAZARDS, GEMINI_MODEL

load_dotenv()

class SafetyAdvisor:
    def __init__(self):
        self.api_key = os.getenv("GEMINI_API_KEY")
        if not self.api_key:
            raise ValueError("GEMINI_API_KEY not found in environment variables")
            
        genai.configure(api_key=self.api_key)
        self.vision_model = genai.GenerativeModel(GEMINI_MODEL)
        
    def analyze_image(self, image):
        """Analyze kitchen image with focus on safety hazards"""
        try:
            # Convert to PIL format for Gemini
            if isinstance(image, np.ndarray):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.open(image)
            
            # Generate comprehensive analysis with hazard focus
            response = self.vision_model.generate_content([
                "As a professional kitchen safety inspector, analyze this image: "
                "1. Identify all safety hazards and hygiene risks "
                "2. Prioritize critical dangers first "
                "3. Provide specific corrective actions "
                "4. Reference standard food safety protocols\n\n"
                "Output format:\n"
                "[SEVERITY] [Issue]: [Description]\n"
                "Action: [Corrective Action]\n"
                "Severity levels: CRITICAL, HIGH, MEDIUM, LOW",
                pil_image
            ])
            
            return response.text
        except Exception as e:
            print(f"Vision analysis error: {str(e)}")
            return "Image analysis service is currently unavailable."
    
    def analyze_video(self, video_path):
        """Analyze entire video using Gemini's video capabilities"""
        try:
            print(f"Uploading video for analysis: {video_path}")
            
            # Upload video file to Gemini
            video_file = genai.upload_file(path=video_path)
            print(f"Video uploaded successfully")
            
            # Wait for video processing
            while video_file.state.name == "PROCESSING":
                print("Waiting for video processing...")
                time.sleep(2)
                video_file = genai.get_file(video_file.name)
            
            if video_file.state.name == "FAILED":
                raise Exception("Video processing failed")
            
            # Generate comprehensive video analysis
            prompt = """
            As a professional kitchen safety inspector, analyze this entire video for safety hazards and hygiene risks.
            Provide a comprehensive report covering:
            
            1. OVERALL SAFETY ASSESSMENT
            - General safety level (CRITICAL/HIGH/MEDIUM/LOW risk)
            - Summary of main concerns
            
            2. SPECIFIC HAZARDS IDENTIFIED
            - Fire hazards and risks
            - Food safety violations
            - Equipment misuse
            - Personal safety issues
            - Hygiene problems
            
            3. TIMELINE OF CRITICAL MOMENTS
            - Approximate timestamps of dangerous situations
            - Immediate risks that require attention
            
            4. CORRECTIVE ACTIONS
            - Priority fixes needed
            - Best practice recommendations
            - Training suggestions
            
            5. COMPLIANCE ASSESSMENT
            - Food safety regulation compliance
            - Kitchen safety standard adherence
            
            Be specific, actionable, and prioritize by severity. Focus on preventing accidents and foodborne illness.
            """
            
            response = self.vision_model.generate_content([
                prompt,
                video_file
            ])
            
            # Clean up uploaded file
            genai.delete_file(video_file.name)
            
            return response.text
            
        except Exception as e:
            print(f"Video analysis error: {str(e)}")
            return f"Video analysis service encountered an error: {str(e)}"
    
    def analyze_frame(self, frame):
        """Analyze individual frame for safety issues"""
        try:
            # Convert to PIL format for Gemini
            if isinstance(frame, np.ndarray):
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(frame_rgb)
            else:
                pil_image = Image.open(frame)
            
            # Generate frame-specific analysis
            response = self.vision_model.generate_content([
                "As a kitchen safety expert, quickly analyze this frame: "
                "1. Identify immediate safety hazards "
                "2. Note any unsafe behaviors or conditions "
                "3. Assess food handling practices "
                "4. Check equipment usage\n\n"
                "Keep response concise but specific. "
                "Focus on actionable safety concerns.",
                pil_image
            ])
            
            return response.text
        except Exception as e:
            print(f"Frame analysis error: {str(e)}")
            return "Frame analysis temporarily unavailable."
    
    def analyze_video_frame(self, frame, previous_insights):
        """
        Analyze video frame with temporal context
        (Backward compatibility method)
        """
        return self.analyze_frame(frame)
    
    def detect_objects(self, image, existing_detections):
        """
        Use Gemini to detect safety-relevant objects not found by local models
        Returns list of additional detections
        """
        try:
            # Convert to PIL format for Gemini
            if isinstance(image, np.ndarray):
                image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(image)
            else:
                pil_image = Image.open(image)
            
            # Get existing class names
            existing_classes = {d["class"].lower() for d in existing_detections}
            
            # Prompt Gemini to detect safety-relevant objects
            prompt = (
                "You are a kitchen safety object detector. Identify ONLY safety-relevant objects "
                f"from this list: {', '.join(SAFETY_HAZARDS)}. "
                "Return ONLY a comma-separated list of detected objects present in the image "
                "that are relevant to kitchen safety. Do not include any other text or explanations."
            )
            
            response = self.vision_model.generate_content([prompt, pil_image])
            detected_objects = [obj.strip().lower() for obj in response.text.split(",")]
            
            # Filter new objects not detected by local models
            new_objects = [
                obj for obj in detected_objects
                if obj in SAFETY_HAZARDS and obj not in existing_classes
            ]
            
            return new_objects
        except Exception as e:
            print(f"Object detection error: {str(e)}")
            return []
    
    def generate_safety_report(self, detections, analysis_results):
        """Generate comprehensive safety report"""
        try:
            hazard_summary = []
            for detection in detections:
                hazard_summary.append(f"- {detection['class']} (confidence: {detection['confidence']:.2f})")
            
            prompt = f"""
            Generate a professional kitchen safety report based on:
            
            DETECTED OBJECTS:
            {chr(10).join(hazard_summary)}
            
            ANALYSIS RESULTS:
            {analysis_results}
            
            Provide:
            1. Executive Summary
            2. Risk Assessment
            3. Immediate Actions Required
            4. Long-term Recommendations
            5. Compliance Status
            
            Format as a professional safety audit report.
            """
            
            response = self.vision_model.generate_content([prompt])
            return response.text
            
        except Exception as e:
            print(f"Report generation error: {str(e)}")
            return "Safety report generation temporarily unavailable."