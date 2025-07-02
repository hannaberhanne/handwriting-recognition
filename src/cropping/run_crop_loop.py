import cv2
import os
from crop_utils import crop_boxes

img_path = "data/processed/cleaned/morphology/page_03.png"
output_dir = "data/processed/cropped/cropped_by_boxes/"
os.makedirs(output_dir, exist_ok=True)

image = cv2.imread(img_path)
if image is None:
    print(f"Could not load image: {img_path}")
else:
    boxes = crop_boxes(image)
    if not boxes:
        print("No boxes detected.")
    else:
        for i, (x, y, w, h) in enumerate(boxes):
            cropped = image[y:y+h, x:x+w]
            save_path = os.path.join(output_dir, f"page_03_box_{i+1:02d}.png")
            cv2.imwrite(save_path, cropped)
            print(f"Saved box {i+1} to {save_path}")