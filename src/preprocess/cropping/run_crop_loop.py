import cv2
import os
from crop_utils import crop_boxes

img_path = "data/processed/cleaned/morphology/page_03.png"
output_dir = "data/processed/cropped/cropped_by_boxes/"
os.makedirs(output_dir, exist_ok=True)

image = cv2.imread(img_path)
if image is None:
    print(f"Could not load image: {img_path}")
    exit()

boxes = crop_boxes(image)
if not boxes:
    print("No boxes detected.")
    exit()

# Group boxes by x-position into columns
column_threshold = 20
columns = []

for box in boxes:
    x, y, w, h = box
    placed = False
    for col in columns:
        if abs(col[0][0] - x) < column_threshold:
            col.append(box)
            placed = True
            break
    if not placed:
        columns.append([box])

# Sort columns left-to-right by x, and each column top-to-bottom by y
columns = sorted(columns, key=lambda col: col[0][0])
for col_idx, col in enumerate(columns, start=1):
    col_sorted = sorted(col, key=lambda b: b[1])  # sort by y
    for row_idx, (x, y, w, h) in enumerate(col_sorted, start=1):
        cropped = image[y:y+h, x:x+w]
        save_path = os.path.join(
            output_dir, f"page_03_col{col_idx:02d}_row{row_idx:02d}.png"
        )
        cv2.imwrite(save_path, cropped)
        print(f"Saved {save_path}")