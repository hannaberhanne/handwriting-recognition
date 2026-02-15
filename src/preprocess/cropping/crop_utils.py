import cv2
import numpy as np
import os
import glob

# Parameters optimized for content capture
BLACK_THRESHOLD = 50
WHITE_THRESHOLD = 200
MIN_CELL_AREA = 800
ROW_GROUP_THRESHOLD = 20
CONTENT_PADDING = 5  # Extra pixels around content

def preprocess_for_boxes(image):
    """Enhance boxes while preserving content"""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY) if len(image.shape) == 3 else image.copy()
    
    # Use adaptive threshold to keep content visible
    thresh = cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV, 21, 10
    )
    
    # Horizontal and vertical kernels to detect box lines
    horizontal_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (20, 1))
    vertical_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 20))
    
    # Detect and enhance lines
    horizontal = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, horizontal_kernel)
    vertical = cv2.morphologyEx(thresh, cv2.MORPH_OPEN, vertical_kernel)
    box_structure = cv2.bitwise_or(horizontal, vertical)
    
    return box_structure

def get_content_roi(image, box):
    """Get region of interest including content with padding"""
    x, y, w, h = box
    # Expand area slightly to ensure content is captured
    x1 = max(0, x - CONTENT_PADDING)
    y1 = max(0, y - CONTENT_PADDING)
    x2 = min(image.shape[1], x + w + CONTENT_PADDING)
    y2 = min(image.shape[0], y + h + CONTENT_PADDING)
    return image[y1:y2, x1:x2]

def detect_and_save_boxes(image_path, output_dir):
    """Main function to detect boxes and save their contents"""
    image = cv2.imread(image_path)
    if image is None:
        print(f"Could not load image: {image_path}")
        return False
    
    # Preprocess to enhance box structure
    box_mask = preprocess_for_boxes(image)
    
    # Find contours of boxes
    contours, _ = cv2.findContours(box_mask, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    
    # Process detected boxes
    boxes = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        if w * h > MIN_CELL_AREA:
            boxes.append((x, y, w, h))
    
    # Group and save box contents
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    os.makedirs(output_dir, exist_ok=True)
    
    for i, box in enumerate(boxes, start=1):
        content_roi = get_content_roi(image, box)
        save_path = os.path.join(output_dir, f"{base_name}_box_{i:03d}.png")
        cv2.imwrite(save_path, content_roi)
    
    return True

if __name__ == "__main__":
    input_dir = "data/processed/cleaned/morphology/"
    output_dir = "data/processed/cropped/cropped_by_boxes/"
    
    for img_path in glob.glob(os.path.join(input_dir, "*.png")):
        print(f"Processing {img_path}...")
        if detect_and_save_boxes(img_path, output_dir):
            print(f"Saved box contents to {output_dir}")
        else:
            print(f"Failed to process {img_path}")