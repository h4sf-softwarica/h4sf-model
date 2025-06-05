#  Kitchen Safety Testing System

This system helps test kitchen safety detection using sample images. It provides:
- Object detection with Roboflow models
- AI-powered safety analysis with Gemini
- Detailed test reports

##  Quick Start Guide

### 1. Setup Test Environment
```bash
# Create project folder
mkdir kitchen-safety-test
cd kitchen-safety-test

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate    # Windows

# Install dependencies
pip install opencv-python numpy roboflow google-generativeai python-dotenv
```

### 2. Create Project Files
```bash
# Create main files
touch main.py model_manager.py visualization.py safety_advisor.py 
touch image_processing.py parser.py

# Create test folder
mkdir test_images
```

### 3. Add Sample Images
Place your test images in the `test_images` folder:
- JPG or PNG format
- Recommended size: 640x480 or similar
- Example kitchen scenes

### 4. Configure API Keys
Create `.env` file:
```env
ROBOFLOW_API_KEY="your_roboflow_api_key"
GEMINI_API_KEY="your_gemini_api_key"
```

### 5. Run Tests
```bash
# Basic test (default settings)
python main.py

# Custom test
python main.py --model kitchen --confidence 0.4 --input my_test_images --output custom_results

# Test without AI analysis
python main.py --no-ai
```

##  Test Results Structure
```
test_results/
├── test_results.json         # Full detection results
├── simplified_results.json   # Top detections per model
└── safety_insights.txt       # AI safety analysis (if enabled)
```

##  Command Reference
| Command                      | Description                          | Default          |
|------------------------------|--------------------------------------|------------------|
| `--mode test`                | Test mode (process images)           | test             |
| `--model <model>`            | meat, kitchen, or sausage            | kitchen          |
| `--confidence <value>`       | Detection threshold (0-1)            | 0.3              |
| `--input <path>`             | Input image folder                   | test_images      |
| `--output <path>`            | Output directory                     | test_results     |
| `--no-ai`                    | Disable Gemini analysis              | False            |
| `--video <path>`             | Video path (for video mode)          | -                |

##  Sample Test Cases

### 1. Basic Safety Check
```bash
python main.py --model kitchen
```

### 2. High Confidence Detection
```bash
python main.py --model meat --confidence 0.5
```

### 3. Custom Test Set
```bash
python main.py --input custom_tests --output high_confidence_results --confidence 0.6
```

### 4. Pure Object Detection
```bash
python main.py --no-ai
```

##  Understanding Results

### test_results.json
- Contains full detection details for all images
- Structure:
  ```json
  [
    {
      "image": "test1.jpg",
      "models": {
        "kitchen": {
          "predictions": [{"class": "knife", ...}],
          "prediction_count": 3,
          "inference_time": 0.45
        }
      },
      "ai_analysis": "Safety analysis text..."
    }
  ]
  ```

### simplified_results.json
- Shows only the highest confidence detection per model
- Structure:
  ```json
  [
    {
      "image": "test1.jpg",
      "models": {
        "kitchen": {
          "class": "knife",
          "confidence": 0.92,
          "bbox": {"x": 100, "y": 200, "w": 50, "h": 50}
        }
      }
    }
  ]
  ```

### safety_insights.txt
- Contains Gemini's safety analysis for each image
- Example:
  ```
  Image: kitchen_scene1.jpg
  Analysis: [CRITICAL] Knife left near edge of counter. Action: Store in knife block immediately.
  [WARNING] Raw meat on wooden cutting board. Action: Use plastic board for meats.
  --------------------------------------------------
  ```

## 💡 Tips for Effective Testing
1. Start with 3-5 representative images
2. Test all models: `--model meat`, `--model kitchen`, `--model sausage`
3. Try different confidence levels (0.2-0.8)
4. Review both JSON and text outputs
5. Compare AI vs non-AI results with `--no-ai` flag
```

### Sample Test Folder Structure
```
kitchen-safety-test/
├── .env
├── main.py
├── model_manager.py
├── visualization.py
├── safety_advisor.py
├── image_processing.py
├── parser.py
├── test_images/
│   ├── kitchen1.jpg
│   ├── kitchen2.jpg
│   ├── meat_prep.jpg
│   └── sausage_station.jpg
└── test_results/  (created after run)
```

### How to Run Test Cases

1. **Basic Test** (Kitchen model, default confidence):
```bash
python main.py
```

2. **Meat Detection Test** (Higher confidence):
```bash
python main.py --model meat --confidence 0.5
```

3. **Custom Test Folder**:
```bash
python main.py --input my_special_tests --output custom_results
```

4. **Without AI Analysis**:
```bash
python main.py --no-ai
```

### Expected Output
```
 Starting test mode with images from: test_images
 Found 4 test images

Processing: kitchen1.jpg
  meat: 2 detections in 0.45s
  kitchen: 3 detections in 0.52s
  sausage: 0 detections in 0.48s
 AI Analysis:
[CRITICAL] Knife left near edge of counter. Action: Store in knife block immediately.
[WARNING] Wet floor with no sign. Action: Place wet floor sign immediately.

Processing: kitchen2.jpg
...

 Test complete!
- Full results: test_results/test_results.json
- Simplified results: test_results/simplified_results.json
- Safety insights: test_results/safety_insights.txt
```

This setup allows  to:
1. Quickly test with sample images
2. Get detailed detection results
3. Receive AI-powered safety insights
4. Customize testing parameters
5. Compare different models and confidence levels
