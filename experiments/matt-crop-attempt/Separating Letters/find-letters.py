import cv2
import numpy as np
import os

BLACK_THRESHOLD = 175
WHITE_THRESHOLD = 200
DELTA = 3
MARGIN = 8

def smooth_white(img):
    num_col = img.shape[1]
    num_row = img.shape[0]

    for r in range(num_row):
        for c in range(num_col):
            if img[r,c,0] > WHITE_THRESHOLD:
                img[r,c,0] = 255
                img[r,c,1] = 255
                img[r,c,2] = 255
            if (r < MARGIN) or (r > num_row - MARGIN):
                img[r,c,0] = 255
                img[r,c,1] = 255
                img[r,c,2] = 255
    return img

#Note: Uses the variable DELTA (global constant)
#      to determine how many lines in a row we need to find
def delta_mixed(col, img):
    num_col = img.shape[1]
    num_row = img.shape[0]

    cur_col = col
    end_col = min(num_col-1, col + DELTA)

    while cur_col <= end_col:
        found_white = False
        found_black = False

        cur_row = 0
        while (cur_row < num_row) and ( not found_white or not found_black):
            if img[cur_row, cur_col, 0] < BLACK_THRESHOLD:
                found_black = True
            if img[cur_row, cur_col, 0] > WHITE_THRESHOLD:
                found_white = True
            cur_row = cur_row + 1

        if (not found_white) or (not found_black):
            return False

        cur_col = cur_col + 1

    return True
        
#Note: Uses the variable DELTA (global constant)
#      to determine how many lines in a row we need to find
def delta_white(col, img):
    num_col = img.shape[1]
    num_row = img.shape[0]

    cur_col = col
    end_col = min(num_col-1, col + DELTA)

    while cur_col <= end_col:
        found_black = False

        cur_row = 0
        while (cur_row < num_row) and ( not found_black):
            if img[cur_row, cur_col, 0] < BLACK_THRESHOLD:
                found_black = True
            cur_row = cur_row + 1

        if found_black:
            return False

        cur_col = cur_col + 1

    return True
    
#Note: col is the col to start looking for a letter
def find_letter(img, col):
    num_col = img.shape[1]
    num_row = img.shape[0]

    start_col = col
    
    #Loop through the columns
    #Find one Delta columns in a row that are a mix of white and black
    for c in range(col,num_col):
        if delta_mixed(c, img):
            start_col = c
            break

    #Now we move forward
    #Until we find Delta columns in a row that are all white
    for c in range(start_col + 1, num_col):
        if delta_white(c, img):
            end_col = c
            break
        
    return (start_col-1, end_col+1)


img_file = "test-word.png"

base_img = cv2.imread(img_file)

white_img = smooth_white(cv2.imread(img_file))


#Find all Letters and Save them
num_col = white_img.shape[1]
cur_col = MARGIN
i = 0
while cur_col < num_col - 1:

    start_col, end_col = find_letter(white_img, cur_col)

    #If we actually found a letter
    if end_col <= num_col - MARGIN:
    
        cropped_img = white_img[ : , start_col : end_col]

        cv2.imwrite(f"test-output{i}.png",cropped_img)

    cur_col = end_col
    i = i+1

        
            
