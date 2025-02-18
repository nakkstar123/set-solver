import os
import cv2
import csv
import time
import imutils
import random
import numpy as np

def order_points(pts):
    """
    Orders a set of 4 points in the following order:
    top-left, top-right, bottom-right, bottom-left.
    """
    rect = np.zeros((4, 2), dtype="float32")
    # The top-left will have the smallest sum,
    # whereas the bottom-right will have the largest sum.
    s = pts.sum(axis=1)
    rect[0] = pts[np.argmin(s)]
    rect[2] = pts[np.argmax(s)]
    
    # The top-right will have the smallest difference,
    # whereas the bottom-left will have the largest difference.
    diff = np.diff(pts, axis=1)
    rect[1] = pts[np.argmin(diff)]
    rect[3] = pts[np.argmax(diff)]
    
    return rect

def preprocess_card_image(image_path, debug=False):
    """
    Pre-processes a card image by:
      1. Isolating the card via thresholding and morphological operations.
      2. Computing a rotated bounding box (via minAreaRect) from the largest bright region.
      3. Applying a perspective transform to get a top-down view.
      4. Correcting lighting using CLAHE.
    
    This approach assumes that the card is the largest white region in the image,
    which is typical since the card has a white background.
    
    Parameters:
        image_path (str): Path to the input image.
        debug (bool): If True, displays intermediate processing steps.
    
    Returns:
        processed_image (np.array): The pre-processed, top-down view of the card.
    """
    # Load the image
    image = cv2.imread(image_path)
    if image is None:
        raise ValueError("Image not found: " + image_path)
    
    orig = image.copy()
    # Resize for faster processing while maintaining aspect ratio.
    ratio = image.shape[0] / 500.0
    image = imutils.resize(image, height=500)
    
    # Convert to grayscale and blur to reduce noise.
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Use Otsu's thresholding to segment bright areas (the card should be bright/white)
    # (If your background is also bright, you might need to tweak this step.)
    ret, thresh = cv2.threshold(blurred, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    
    # Optionally invert the threshold if needed:
    # thresh = cv2.bitwise_not(thresh)
    
    # Apply morphological closing to fill small holes and join the white regions.
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    if debug:
        cv2.imshow("Thresholded & Closed", closed)
        cv2.waitKey(0)
    
    # Find external contours in the thresholded image.
    cnts = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = imutils.grab_contours(cnts)
    if len(cnts) == 0:
        raise Exception("No contours found in thresholded image!")
    
    # Assume the largest contour is the card.
    card_contour = max(cnts, key=cv2.contourArea)
    
    # Get a rotated bounding box of the card.
    rect = cv2.minAreaRect(card_contour)  # returns ((center_x, center_y), (width, height), angle)
    box = cv2.boxPoints(rect)             # returns 4 points of the rotated rectangle
    box = np.array(box, dtype="float32")
    
    # Order the points and scale them back to the original image dimensions.
    ordered_box = order_points(box) * ratio
    
    # Compute the width and height of the new (unwarped) image.
    (tl, tr, br, bl) = ordered_box
    widthA = np.linalg.norm(br - bl)
    widthB = np.linalg.norm(tr - tl)
    maxWidth = max(int(widthA), int(widthB))
    
    heightA = np.linalg.norm(tr - br)
    heightB = np.linalg.norm(tl - bl)
    maxHeight = max(int(heightA), int(heightB))
    
    # Destination points for the perspective transform.
    dst = np.array([
        [0, 0],
        [maxWidth - 1, 0],
        [maxWidth - 1, maxHeight - 1],
        [0, maxHeight - 1]
    ], dtype="float32")
    
    # Compute the perspective transform matrix and apply it.
    M = cv2.getPerspectiveTransform(ordered_box, dst)
    warped = cv2.warpPerspective(orig, M, (maxWidth, maxHeight))
    
    # Lighting correction using CLAHE.
    lab = cv2.cvtColor(warped, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    processed_image = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    
    if debug:
        cv2.imshow("Warped", warped)
        cv2.imshow("Processed", processed_image)
        cv2.waitKey(0)
    
    return processed_image



def detect_count(image_path):
    image = preprocess_card_image(image_path)
    # image = cv2.imread(image_path)

    # Crop to remove background
    crop_size = 25
    image = image[crop_size:-crop_size, crop_size:-crop_size]

    # Convert to grayscale
    gray_scaled = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Detect edges
    edged = cv2.Canny(gray_scaled, 100, 250)

    # Apply thresholds
    thresh = cv2.threshold(edged, 100, 255, cv2.THRESH_BINARY)[1]

    # Use morphological closing to fill in gaps
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)

    # Detect contours
    contours = cv2.findContours(closed.copy(), cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    contours = imutils.grab_contours(contours)

    # Choose one contour to crop around; here we choose the largest contour
    largest_contour = max(contours, key=cv2.contourArea)
    x, y, w, h = cv2.boundingRect(largest_contour)

    cv2.drawContours(image, contours, -1, (0, 0, 0))
    cv2.imshow(".", image)
    cv2.waitKey(0)
    
    # Add padding while ensuring we stay within image boundaries
    pad = 10
    x_start = max(0, x - pad)
    y_start = max(0, y - pad)
    x_end = min(image.shape[1], x + w + pad)
    y_end = min(image.shape[0], y + h + pad)
    
    # Return image cropped to a contour
    cropped = image[y_start:y_end, x_start:x_end]
    largest_contour = largest_contour - np.array([[x_start, y_start]])

    return len(contours), cropped, largest_contour








# Example usage (uncomment the following lines to test the function):
if __name__ == "__main__":
    # DATA_PATH = os.path.join(os.getcwd(), "cards_dataset.csv")
    # with open(DATA_PATH) as data_csv:
    #     reader = csv.reader(data_csv)
    #     rows = list(reader)  # Load all rows into a list
    #     random_line = random.choice(rows)
    # IMAGE_PATH = random_line[0]
    # for n in range(1):
    #     card = f"card_{n}.png"
    #     IMAGE_PATH = os.path.join(os.getcwd(), card)
    #     image = preprocess_card_image(card, debug=True)
    #     # count, _ = detect_count(IMAGE_PATH)
    #     # print(f"{card}: {count}")
    # Pick random image
    for _ in range(5):
        IMAGES_DIR = os.path.join(os.getcwd(), "outputs2")
        RANDOM_FOLDER = random.choice(os.listdir(IMAGES_DIR))
        PATH_TO_FOLDER = os.path.join(IMAGES_DIR, RANDOM_FOLDER)
        RANDOM_IMAGE = random.choice(os.listdir(PATH_TO_FOLDER))
        IMAGE_PATH = os.path.join(PATH_TO_FOLDER, RANDOM_IMAGE)

        # print(IMAGE_PATH)
        # image = preprocess_card_image(IMAGE_PATH)

        # Processing steps
        count, cropped, contours = detect_count(IMAGE_PATH)
        print(f"Count: {count}")
        # color = detect_color_by_hue(cropped)
        # print(f"{count} x {color}")
        cv2.imshow(".", cropped)
        cv2.waitKey(0)