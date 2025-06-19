"""
Uses morphological operations to reduce background noise in scanned pages.
Produces cleaned grayscale images by subtracting the estimated background.
"""
import cv2
import numpy as np
import os
import glob


# Set input and output directories
input_dir = "data/raw/ahafo_region/"
output_dir = "data/processed/cleaned/morphology/"
os.makedirs(output_dir, exist_ok=True)

import glob
# Process each PNG page image in the input directory
for page_path in glob.glob(os.path.join(input_dir, "*.png")):
    filename = os.path.basename(page_path)
    base = os.path.splitext(filename)[0]
    # Load image and convert to grayscale
    img = cv2.imread(page_path)
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Apply Gaussian blur to smooth the image
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    # Use morphological closing to estimate background
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    bg = cv2.morphologyEx(blurred, cv2.MORPH_CLOSE, kernel)
    # Subtract background from original grayscale image, invert to get white background and dark ink
    diff = 255 - cv2.absdiff(gray, bg)
    diff = cv2.normalize(diff, None, 0, 255, cv2.NORM_MINMAX)

    # Save the cleaned image to the output directory
    cv2.imwrite(os.path.join(output_dir, f"{base}.png"), diff)

print("Morphology-based background subtraction complete.")