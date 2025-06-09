"""
Applies CLAHE (Contrast Limited Adaptive Histogram Equalization) to enhance contrast
on scanned handwritten pages. Saves enhanced grayscale images for OCR preprocessing.
"""
import cv2
import numpy as np
import os
import glob

# Set up input and output directories
input_dir = "data/raw/ahafo_region/"
output_dir = "data/processed/cleaned/clahe/"
os.makedirs(output_dir, exist_ok=True)

# Loop through all PNG page images in the input directory
for page_path in glob.glob(os.path.join(input_dir, "page_*.png")):
    filename = os.path.basename(page_path)
    base = os.path.splitext(filename)[0]
    img = cv2.imread(page_path)
    # Convert the image to grayscale
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Create CLAHE object with clip limit and tile grid size
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # Apply CLAHE and save result to output directory
    clahe_img = clahe.apply(gray)
    cv2.imwrite(os.path.join(output_dir, f"{base}.png"), clahe_img)

print("CLAHE contrast enhancement complete.")