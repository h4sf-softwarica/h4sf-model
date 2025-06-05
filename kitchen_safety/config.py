# kitchen_safety/config.py
SAFETY_HAZARDS = {
    "knife", "fire", "flame", "raw_meat", "raw_chicken", "raw_fish",
    "chemical", "spill", "slippery", "contamination", "bacteria",
    "unrefrigerated", "unsanitary", "mold", "pest", "rodent", "insect"
}

# Threshold for considering a detection valid
DETECTION_THRESHOLD = 0.5

# Gemini model configuration
GEMINI_MODEL = "gemini-1.5-flash"