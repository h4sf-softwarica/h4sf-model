# h4sf Model

## Overview

This repository contains a real-time safety compliance tracker using YOLOv8 for detecting protective gear like hairnets and gloves via webcam. It helps monitor workplace safety by visually identifying key safety equipment in video streams.
 

## Features

- Real-time object detection using YOLOv8 model.
- Detects safety equipment such as hairnets and gloves.
- Highlights detected items with bounding boxes and confidence scores.
- Live video feed from webcam for continuous monitoring.
- Configurable classes and detection confidence threshold.

## Tools and Technology used

- Python 3.13
- OpenCV for real-time video processing
- Ultralytics YOLOv8 for object detection
- PyTorch as the deep learning framework
- Webcam for live video input

## Getting Started

### Platforms

This project supports the following platforms:

- Windows
- Linux
- macOS

### Requirements

- Python 3.7 or higher
- OpenCV
- PyTorch
- Ultralytics YOLOv8 package
- Webcam connected to your system
- **Microsoft Visual C++ Redistributable** (Download from [here](https://aka.ms/vs/16/release/vc_redist.x64.exe)) — required for PyTorch on Windows

### Installation

1. **Clone the repository:**
    ```sh
    git clone https://github.com/h4sf-softwarica/h4sf-model.git
    cd h4sf-model
    ```

2. **Set up a virtual environment (optional but recommended):**
    ```sh
    python -m venv venv
    venv\Scripts\activate          # Windows
    source venv/bin/activate       # Linux/macOS
    ```

3. **Install required packages:**
    ```sh
    pip install -r requirements.txt
    ```

## Usage

- Run the main script:
    ```sh
    python testing.py
    ```
- The program will open your webcam and display real-time detection of specified safety equipment.
- Press 'q' to quit the application.


## Contributing

Contributions are welcome! Please fork the repo, make changes, and submit pull requests. Open issues for bugs or feature requests.
