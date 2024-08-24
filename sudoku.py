# -*- coding: utf-8 -*-
"""
Created on Thu Aug 22 14:57:32 2024

@author: gaspa
"""

import cv2
import numpy as np

def preprocess_image(image_path):
    # Load the image in grayscale
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    
    # Apply Gaussian blur to reduce noise and improve edge detection
    blurred = cv2.GaussianBlur(image, (7, 7), 3)
    
    # Use adaptive thresholding to binarize the image
    thresh = cv2.adaptiveThreshold(blurred, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                   cv2.THRESH_BINARY_INV, 11, 2)
    
    # Optionally, use morphology to make the digits more distinct
    kernel = np.array([[0, 1, 0],
                       [1, 1, 1],
                       [0, 1, 0]], dtype=np.uint8)
    thresh = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    return thresh



def extract_grid(thresh):
    # Find contours
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Find the largest contour (should be the Sudoku grid)
    largest_contour = max(contours, key=cv2.contourArea)
    
    # Approximate the contour to a polygon and get bounding box
    peri = cv2.arcLength(largest_contour, True)
    approx = cv2.approxPolyDP(largest_contour, 0.02 * peri, True)
    
    # If our approximated contour has four points, we can assume it's a square
    if len(approx) == 4:
        points = approx.reshape(4, 2)
    
    return points

def perspective_transform(image, points):
    # Points should be ordered top-left, top-right, bottom-right, bottom-left
    rect = order_points(points)
    
    (tl, tr, br, bl) = rect
    
    # Compute the width and height of the new image
    widthA = np.sqrt(((br[0] - bl[0]) ** 2) + ((br[1] - bl[1]) ** 2))
    widthB = np.sqrt(((tr[0] - tl[0]) ** 2) + ((tr[1] - tl[1]) ** 2))
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.sqrt(((tr[0] - br[0]) ** 2) + ((tr[1] - br[1]) ** 2))
    heightB = np.sqrt(((tl[0] - bl[0]) ** 2) + ((tl[1] - bl[1]) ** 2))
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points are a straight top-down view of the grid
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]], dtype="float32")
    
    # Compute the perspective transform matrix and apply it
    M = cv2.getPerspectiveTransform(rect, dst)
    warped = cv2.warpPerspective(image, M, (maxWidth, maxHeight))
    
    return warped

def order_points(pts):
    # Order points based on their x and y coordinates
    rect = np.zeros((4, 2), dtype="float32")
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

import pytesseract

def extract_digits(grid_image):
    cells = []
    size = grid_image.shape[0] // 9
    
    for i in range(9):
        row = []
        for j in range(9):
            x, y = j * size, i * size
            cell = grid_image[y:y+size, x:x+size]
            
            # Show the cell image for debugging
            #cv2.imshow(f'Cell ({i},{j})', cell)
            #cv2.waitKey(0)  # Press any key to close the image window
            
            digit = extract_digit(cell)
            row.append(digit)
        cells.append(row)
    
    #cv2.destroyAllWindows()
    return cells




def is_valid(board, row, col, num):
    # Check if the number is not already in the row, column, or 3x3 grid
    for i in range(9):
        if board[row][i] == num or board[i][col] == num:
            return False
    
    box_row_start = (row // 3) * 3
    box_col_start = (col // 3) * 3
    
    for i in range(3):
        for j in range(3):
            if board[box_row_start + i][box_col_start + j] == num:
                return False
    
    return True

def solve_sudoku(board):
    for i in range(9):
        for j in range(9):
            if board[i][j] == 0:
                for num in range(1, 10):
                    if is_valid(board, i, j, num):
                        board[i][j] = num
                        if solve_sudoku(board):
                            return True
                        board[i][j] = 0
                return False
    return True

def print_board(board):
    for i in range(9):
        row = ""
        for j in range(9):
            if j == 3 or j == 6:
                row += " | "
            row += str(board[i][j]) + " "
        if i == 3 or i == 6:
            print("-" * 21)
        print(row)


def extract_digit(cell):
    # Convert the cell to grayscale if it's not already
    if len(cell.shape) > 2:
        cell = cv2.cvtColor(cell, cv2.COLOR_BGR2GRAY)
    
    # Resize the cell image to ensure Tesseract works well
    cell = cv2.resize(cell, (32, 32))
    
    # Apply thresholding to get a binary image
    cell = cv2.threshold(cell, 0, 255, cv2.THRESH_BINARY_INV | cv2.THRESH_OTSU)[1]
    
    # Use OCR to extract the digit from the cell
    text = pytesseract.image_to_string(cell, config='--psm 10 --oem 3 -c tessedit_char_whitelist=0123456789')
    
    # Clean the OCR output by removing any non-digit characters
    text = text.strip().replace("\n", "")
    
    print(f'reading row: "{text}"')
    
    # Check if the cleaned text is a digit
    return int(text) if text.isdigit() else 0

import random

def print_board_with_hint(board):
    # Generate a list of all non-zero cells
    non_zero_cells = [(i, j) for i in range(9) for j in range(9) if board[i][j] != 0]
    
    # Randomly select one cell to hint
    if not non_zero_cells:
        print("No numbers to hint.")
        return
    
    hint_row, hint_col = random.choice(non_zero_cells)
    
    # ANSI escape codes for color
    RED = '\033[91m'
    RESET = '\033[0m'
    
    # Print the board with the hint
    for i in range(9):
        row = ""
        for j in range(9):
            if j == 3 or j == 6:
                row += " | "
            if (i == hint_row and j == hint_col):
                row += RED + str(board[i][j]) + RESET + " "
            else:
                row += str(board[i][j]) + " "
        if i == 3 or i == 6:
            print("-" * 21)
        print(row)

def solve_sudoku_from_image(image_path):
    thresh = preprocess_image(image_path)
    points = extract_grid(thresh)
    warped = perspective_transform(cv2.imread(image_path), points)
    
    # Resize to ensure proper OCR
    resized = cv2.resize(warped, (450, 450))
    
    # Extract digits from the image
    board = extract_digits(resized)
    
    # Solve the Sudoku
    if solve_sudoku(board):
        print("Solved Sudoku:")
        print_board(board)
    else:
        print("No solution exists for the given Sudoku.")    
        
    #print_board_with_hint(board)
    return board

#pytesseract.pytesseract.tesseract_cmd = 'C:/OCR/Tesseract-OCR/tesseract.exe'  # your path may be different
pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'


# Example usage
solved_board = solve_sudoku_from_image('sudoku.jpg')
#print(solved_board)
