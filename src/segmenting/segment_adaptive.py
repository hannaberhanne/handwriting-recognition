"""
Applies adaptive thresholding to scanned page images from the ahafo_region dataset.
Outputs binary images to the adaptive cleaning folder for OCR-ready input.
"""
import cv2
import numpy as np
import os
import glob


# Set input and output directories
input_dir = "data/raw/ahafo_region/"
output_dir = "data/processed/cleaned/adaptive/"
os.makedirs(output_dir, exist_ok=True)

# Loop through each page image in the input folder
for page_path in glob.glob(os.path.join(input_dir, "page_*.png")):
    filename = os.path.basename(page_path)
    base = os.path.splitext(filename)[0]
    # Load image and convert to grayscale
    img = cv2.imread(page_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply adaptive thresholding to enhance text
    adaptive = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY,
        11, 2
    )
    # Save the processed binary image
    cv2.imwrite(os.path.join(output_dir, f"{base}.png"), adaptive)

print("Adaptive thresholding complete.")