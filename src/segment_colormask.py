"""
Extracts red and blue ink areas from scanned pages using HSV color masks.
Saves the masked output to the colormask folder for further processing.
"""
import cv2
import numpy as np
import os
import glob

# Define where to read input images from and where to save results
input_dir = "data/raw/ahafo_region/"
output_dir = "data/processed/cleaned/colormask/"
os.makedirs(output_dir, exist_ok=True)

 # Loop through each page image in the input folder
for page_path in glob.glob(os.path.join(input_dir, "page_*.png")):
    filename = os.path.basename(page_path)
    base = os.path.splitext(filename)[0]
    # Load the image and convert to HSV color space
    img = cv2.imread(page_path)
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

    # Define HSV ranges for red and blue ink
    red_lower = (0, 70, 50)
    red_upper = (10, 255, 255)
    blue_lower = (90, 50, 50)
    blue_upper = (130, 255, 255)

    # Create masks for red and blue areas
    red_mask = cv2.inRange(hsv, red_lower, red_upper)
    blue_mask = cv2.inRange(hsv, blue_lower, blue_upper)
    combined_mask = cv2.bitwise_or(red_mask, blue_mask)
    # Apply the combined color mask to isolate colored ink
    result = cv2.bitwise_and(img, img, mask=combined_mask)

    # Save the masked image to the output folder
    cv2.imwrite(os.path.join(output_dir, f"{base}.png"), result)

print("Color masking complete (red + blue).")