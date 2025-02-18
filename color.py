import os
import cv2
import csv
import time
import imutils
import random
import numpy as np

from count import detect_count

def detect_color(cropped_image, sat_thresh=50):
    """
    Identifies the color of the symbol in the cropped image.
    
    This function converts the image to the HSV color space and creates a mask to
    filter out near-white pixels (likely background). Then, it computes the average BGR
    color of the remaining pixels and compares this average with predefined ground truth
    color values for red, purple, and green.
    
    Parameters:
        cropped_image (np.array): Cropped image containing one symbol.
        sat_thresh (int): Saturation threshold to consider a pixel as "colored" (default=50).
        
    Returns:
        str: Detected color, one of 'red', 'purple', or 'green'.
    """
    # Convert the image to HSV to better isolate colored pixels.
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    # Create a mask based on saturation. Pixels with low saturation are likely white/background.
    mask = hsv[..., 1] > sat_thresh
    
    # If no pixels meet the criteria, fall back to using the whole image.
    if np.count_nonzero(mask) == 0:
        mask = np.ones(hsv.shape[:2], dtype=bool)
    
    # Calculate the average BGR color from the masked (colored) pixels.
    # cropped_image[mask] returns an array of shape (num_pixels, 3)
    avg_color = np.mean(cropped_image[mask], axis=0)
    
    # Define ground truth color values in BGR (as OpenCV uses BGR).
    # These values might need to be calibrated for your dataset.
    ground_truth = {
        'red': (0, 0, 255),
        'purple': (97, 4, 184),
        'green': (1, 161, 12)
    }

    # Define weights for B, G, R channels; here we weight blue more heavily.
    weights = np.array([3, 1, 1])  # B, G, R weights
    
    distances = {}
    for color, gt_color in ground_truth.items():
        diff = avg_color - gt_color
        distances[color] = np.sqrt(np.sum((weights * diff) ** 2))
    
    # Choose the color with the smallest distance.
    detected_color = min(distances, key=distances.get)
    
    return detected_color

def detect_color_lab(cropped_image, sat_thresh=50):
    """
    Identifies the color of the symbol in the cropped image by computing the average BGR color,
    converting it to LAB space, and comparing it to ground truth colors (also in LAB space)
    using Euclidean distance (ΔE).
    
    Parameters:
        cropped_image (np.array): Cropped image containing one symbol.
        sat_thresh (int): Saturation threshold in HSV to consider a pixel as "colored" (default=50).
        
    Returns:
        str: Detected color, one of 'red', 'purple', or 'green'.
    """
    # Convert the image to HSV to create a mask that excludes low-saturation (background) pixels.
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    mask = hsv[..., 1] > sat_thresh
    if np.count_nonzero(mask) == 0:
        mask = np.ones(hsv.shape[:2], dtype=bool)
    
    # Compute the average color (in BGR) for the masked pixels.
    avg_color_bgr = np.mean(cropped_image[mask], axis=0)
    
    # Convert the average BGR color to LAB.
    avg_color_bgr_uint8 = np.uint8([[avg_color_bgr]])
    avg_color_lab = cv2.cvtColor(avg_color_bgr_uint8, cv2.COLOR_BGR2LAB)[0][0]
    
    # Define ground truth colors in BGR.
    # ground_truth_bgr = {
    #     'red': np.array([0, 0, 255]),
    #     'purple': np.array([128, 0, 128]),
    #     'green': np.array([0, 255, 0])
    # }
    ground_truth_bgr = {
        'red': (0, 0, 255),
        'purple': (97, 4, 184),
        'green': (1, 161, 12)
    }
    
    # Convert ground truth colors from BGR to LAB.
    ground_truth_lab = {}
    for color, bgr in ground_truth_bgr.items():
        bgr_uint8 = np.uint8([[bgr]])
        lab = cv2.cvtColor(bgr_uint8, cv2.COLOR_BGR2LAB)[0][0]
        ground_truth_lab[color] = lab
    
    # Compute Euclidean distances (ΔE) in LAB space.
    distances = {}
    for color, lab in ground_truth_lab.items():
        diff = avg_color_lab.astype("float") - lab.astype("float")
        distances[color] = np.sqrt(np.sum(diff**2))
    
    detected_color = min(distances, key=distances.get)
    return detected_color

def overlay_detected_color(cropped_image, contours, detected_color, thickness=2, alpha=0.5):
    """
    Overlays the detected color on the contours of the cropped image.
    
    Parameters:
        cropped_image (np.array): The cropped image containing the symbol.
        contours (list): List of contours from Step 1.
        detected_color (str): The detected color as a string ('red', 'purple', or 'green').
        thickness (int): Thickness of the contour lines.
        alpha (float): Blending factor for overlay.
    
    Returns:
        np.array: Image with colored contour overlay.
    """
    # Map color names to BGR values
    color_map = {
        'red': (0, 0, 255),
        'purple': (97, 4, 184),
        'green': (1, 161, 12)
    }
    
    # Retrieve the BGR color for the detected color
    overlay_color = color_map.get(detected_color, (255, 255, 255))
    
    # Create a copy of the cropped image to draw the overlay
    overlay = cropped_image.copy()
    
    # Draw the contours on the overlay image
    cv2.drawContours(overlay, contours, -1, overlay_color, thickness)
    
    # Blend the overlay with the original cropped image for a transparent effect
    # output = cv2.addWeighted(overlay, alpha, cropped_image, 1 - alpha, 0)

    return overlay

def detect_color_by_hue(cropped_image, sat_thresh=50):
    """
    Identifies the dominant color of the symbol by analyzing the hue channel
    of the cropped image. This method is often more robust for differentiating
    similar colors (like red vs. purple) since it uses the mode of the hue distribution.
    
    Parameters:
        cropped_image (np.array): Cropped image containing one symbol.
        sat_thresh (int): Saturation threshold to filter out background pixels.
    
    Returns:
        str: Detected color: 'red', 'purple', or 'green'.
    """
    # Convert to HSV
    hsv = cv2.cvtColor(cropped_image, cv2.COLOR_BGR2HSV)
    
    # Create a mask that keeps only pixels with a saturation above the threshold
    mask = hsv[..., 1] > sat_thresh
    if np.count_nonzero(mask) == 0:
        mask = np.ones(hsv.shape[:2], dtype=bool)
    
    # Extract the hue channel values of the masked (colored) pixels.
    hue_vals = hsv[..., 0][mask]
    
    # Compute the histogram of hue values. We'll use 180 bins (one for each degree in OpenCV's scale)
    hist, bin_edges = np.histogram(hue_vals, bins=180, range=(0,180))
    
    # The dominant hue is the center of the bin with the highest count.
    dominant_bin = np.argmax(hist)
    dominant_hue = (bin_edges[dominant_bin] + bin_edges[dominant_bin+1]) / 2.0

    # Debug print (optional): Uncomment next line to see dominant hue
    # print("Dominant hue:", dominant_hue)
    
    # Define thresholds (these can be tuned based on your images):
    # - Red typically appears near 0 (or near 180)
    # - Green typically appears around 35 to 85
    # - Purple falls in between these ranges.
    if dominant_hue < 10 or dominant_hue > 170:
        return 'red'
    elif 35 <= dominant_hue <= 85:
        return 'green'
    else:
        return 'purple'

if __name__ == "__main__":
    # DATA_PATH = os.path.join(os.getcwd(), "cards_dataset.csv")
    # with open(DATA_PATH) as data_csv:
    #     reader = csv.reader(data_csv)
    #     rows = list(reader)  # Load all rows into a list
    #     random_line = random.choice(rows)
    # IMAGE_PATH = random_line[0] 
    for _ in range(5):
        # Pick random image
        IMAGES_DIR = os.path.join(os.getcwd(), "outputs2")
        RANDOM_FOLDER = random.choice(os.listdir(IMAGES_DIR))
        PATH_TO_FOLDER = os.path.join(IMAGES_DIR, RANDOM_FOLDER)
        RANDOM_IMAGE = random.choice(os.listdir(PATH_TO_FOLDER))
        IMAGE_PATH = os.path.join(PATH_TO_FOLDER, RANDOM_IMAGE)

        print(IMAGE_PATH)

        # Processing steps
        count, cropped, contours = detect_count(IMAGE_PATH)
        color = detect_color_by_hue(cropped)
        print(f"{count} x {color}")
        # cv2.imshow(".", cropped)
        # cv2.waitKey(0)
