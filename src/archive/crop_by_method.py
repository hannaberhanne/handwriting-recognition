import cv2
import numpy as np
import os

methods = ["adaptive", "clahe", "colormask", "morphology"]
input_root = "data/processed/cleaned"
output_root = "data/processed/cropped"

# going through each method folder to process images
for method in methods:
    input_folder = os.path.join(input_root, method)
    output_folder = os.path.join(output_root, method)
    os.makedirs(output_folder, exist_ok=True)

    print(f"\nProcessing images for method: {method}")

    for page_num in range(1, 8):  # pages 1 through 7
        file_name = f"page_0{page_num}.png"
        img_file = os.path.join(input_folder, file_name)
        save_path = os.path.join(output_folder, file_name)

        if not os.path.exists(img_file):
            print(f"  Warning: File not found - {img_file}")
            continue

        image = cv2.imread(img_file)
        if image is None:
            print(f"  Error: Could not read image - {img_file}")
            continue

        gray_img = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        col_sum = np.sum(gray_img < 200, axis=0)
        print("Here is the Shape of gray_img", gray_img.shape)

        left_edge = 0
        for x_pos, val in enumerate(col_sum):
            if val > gray_img.shape[0] * 0.05:
                left_edge = max(0, x_pos - 10)
                break

        right_edge = gray_img.shape[1]
        for x_pos in range(gray_img.shape[1] - 1, left_edge, -1):
            if col_sum[x_pos] > gray_img.shape[0] * 0.05:
                right_edge = min(gray_img.shape[1], x_pos + 10)
                break

        cropped_img = image[:, left_edge:right_edge]
        cv2.imwrite(save_path, cropped_img)
        print(f"  Done saving cropped image to: {save_path}")