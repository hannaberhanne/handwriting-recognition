import cv2
import numpy as np
import os
import glob

# Input & output setup
input_dir = "data/raw/example_page/"
cleaned_dir = "data/processed/cleaned/"
cropped_dir = "data/processed/cropped/"
os.makedirs(cleaned_dir, exist_ok=True)
os.makedirs(cropped_dir, exist_ok=True)

# Loop through all page images
for page_path in glob.glob(os.path.join(input_dir, "page_*.png")):
    filename = os.path.basename(page_path)
    base = os.path.splitext(filename)[0]

    print(f"\nProcessing {filename}...")

    # Load image
    img = cv2.imread(page_path)
    if img is None:
        print(f"Could not load {page_path}")
        continue

    # Light denoise + grayscale + normalization
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
    gray = cv2.cvtColor(denoised, cv2.COLOR_BGR2GRAY)
    norm = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)

    # Threshold softly to isolate text-ish areas
    _, binary = cv2.threshold(norm, 180, 255, cv2.THRESH_BINARY_INV)

    # Light dilation to merge components
    kernel = np.ones((2, 2), np.uint8)
    binary = cv2.dilate(binary, kernel, iterations=1)

    # Invert mask → everything else to white
    mask_inv = cv2.bitwise_not(binary)
    img[mask_inv > 0] = [255, 255, 255]

    # Save cleaned image
    cleaned_path = os.path.join(cleaned_dir, f"{base}_cleaned.png")
    cv2.imwrite(cleaned_path, img)
    print(f"→ Saved cleaned to {cleaned_path}")

    # Column crop using projection profile
    gray_clean = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    proj = np.sum(gray_clean < 200, axis=0)

    left = 0
    for i, val in enumerate(proj):
        if val > gray_clean.shape[0] * 0.05:
            left = max(0, i - 10)
            break

    right = gray_clean.shape[1] // 2
    for i in range(left, right):
        if proj[i] < 5:
            right = i + 10
            break

    # Crop and save
    crop = img[:, left:right]
    crop_path = os.path.join(cropped_dir, f"{base}_cropped.png")
    cv2.imwrite(crop_path, crop)
    print(f"→ Saved crop to {crop_path}")

print("\n All pages processed.")