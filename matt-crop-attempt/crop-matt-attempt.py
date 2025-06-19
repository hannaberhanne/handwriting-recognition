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

# Tries to find a black box in the image for cropping
#  ... Starts looking at start_row, start_col
#  ... Searches down and right from the starting position
# Returns the subset of the image that contains the box
def find_box(start_row, start_col, img):

    num_row = image.shape[0]
    num_col = image.shape[1]

    ### Let's find the upper-left corner of a box of text

    upper_left_r = start_row
    upper_left_c = start_col

    for r in range(start_row,num_row):
        if row_both_white_black(r, image):
            c = find_black_col(r, start_col,image)
            if white_both_side(r, c, image):
                print(f"Found Upper-Left -- Row: {r}, Col: {c}")
                upper_left_r = r
                upper_left_c = c
                break


    ### Now Let's find the lower right corner

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
    return cropped_img

###################################
## START OF MAIN PROGRAM
###################################


img_file = "page_02.png"

image = cv2.imread(img_file)

# Printing an RGB value just as a sanity check
#print(image[1,1,:])

cropped_img = find_box(500,500,image)

save_file_name = "test_crop.png"
cv2.imwrite(save_file_name, cropped_img)
print(f"  Done saving cropped image to: {save_file_name}")

