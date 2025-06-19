import cv2
import numpy as np
import os

BLACK_THRESHOLD = 100
WHITE_THRESHOLD = 150
DELTA = 6

# Determines whether a row in the image has both black and white pixels
def row_both_white_black(row, img):
    num_col = img.shape[1]
    found_black = False
    found_white = False
    for c in range(num_col):
        if img[row,c,0] <= BLACK_THRESHOLD:
            found_black = True
        if img[row,c,0] >= WHITE_THRESHOLD:
            found_white = True
        if found_black and found_white:
            return True
        
    #If it makes it through the loop without finding white and black
    # Then return False
    return False

# Finds the first column number where there is Black in the given Row
#   ... starts looking at column number start_col
def find_black_col(row, start_col, img):
    num_col = img.shape[1]
    for c in range(start_col, num_col):
        if img[row,c,0] <= BLACK_THRESHOLD:
            return c
    # If you don't find a suitable column, return -1 as a signal
    return -1

# Returns True if there is white on both sides of a given Black column
#   ... in a particular row
# This is used to detect a vertical black line
def white_both_side(row, col, img):
    found_white_left_side = False
    found_white_right_side = False
    for delta in range(-6, 6):
        if img[row, col + delta, 0] >= WHITE_THRESHOLD:
            if delta < 0:
                found_white_left_side = True
            if delta > 0:
                found_white_right_side = True
    if found_white_left_side and found_white_right_side:
        return True
    else:
        return False

img_file = "page_02.png"

image = cv2.imread(img_file)

num_row = image.shape[0]
num_col = image.shape[1]

# Printing an RGB value just as a sanity check
#print(image[1,1,:])

################
# Let's find the upper-left corner of a box of text

upper_left_r = 0
upper_left_c = 0

for r in range(0,num_row):
    if row_both_white_black(r, image):
        c = find_black_col(r, 0,image)
        if white_both_side(r, c, image):
            print(f"Row: {r}, Col: {c}")
            upper_left_r = r
            upper_left_c = c
            break

################
# Now Let's find the lower right corner

new_row = upper_left_r + DELTA

found_right_side_of_box = False
new_col = upper_left_c + DELTA - 1

while not found_right_side_of_box:
    new_col = find_black_col(new_row, new_col, image)
    if white_both_side(new_row, new_col, image):
        found_right_side_of_box = True

# Go down the image until you find a solid black horizontal line
while white_both_side(new_row, new_col, image):
    new_row = new_row + 1

lower_right_r = new_row
lower_right_c = new_col


cropped_img = image[upper_left_r : lower_right_r, upper_left_c : lower_right_c, :]
print(f"Shape of Cropped Image: {cropped_img.shape}")


save_file_name = "test_crop.png"
cv2.imwrite(save_file_name, cropped_img)
print(f"  Done saving cropped image to: {save_file_name}")


'''
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

        left_edge = 0
        for x_pos, val in enumerate(col_sum):
            if val > gray_img.shape[0] * 0.05:
                left_edge = max(0, x_pos - 10)
                break

        right_edge = gray_img.shape[1] // 2
        for x_pos in range(left_edge, gray_img.shape[1] // 2):
            if col_sum[x_pos] < 5:
                right_edge = x_pos + 10
                break

        cropped_img = image[:, left_edge:right_edge]
        cv2.imwrite(save_path, cropped_img)
        print(f"  Done saving cropped image to: {save_path}")

'''
