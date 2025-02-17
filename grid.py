import cv2
import numpy as np
import os
from glob import glob

def extract_cards_3x4(image_path):
    # 1. Read and preprocess
    src = cv2.imread(image_path)
    gray = cv2.cvtColor(src, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray, (5,5), 0)
    edges = cv2.Canny(blurred, 50, 150)

    # 2. Find contours and filter
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    candidate_rects = []
    for cnt in contours:
        x, y, w, h = cv2.boundingRect(cnt)
        aspect_ratio = float(w)/h
        area = w*h
        # Filter: assume typical card aspect ratio ~ 0.7, area in some range, etc.
        if 0.4 < aspect_ratio < 0.8 and area > 10000:
            candidate_rects.append((x,y,w,h))

    # Sort by y first
    candidate_rects.sort(key=lambda r: r[1])

    # 3. Group rectangles into rows by y-coordinate
    rows = []
    current_row = []
    row_threshold = 50  # tweak this for your images
    last_y = None

    for rect in candidate_rects:
        x, y, w, h = rect
        if last_y is None:
            current_row.append(rect)
            last_y = y
        else:
            if abs(y - last_y) < row_threshold:
                current_row.append(rect)
            else:
                rows.append(current_row)
                current_row = [rect]
            last_y = y

    if current_row:
        rows.append(current_row)

    # Within each row, sort by x
    for row in rows:
        row.sort(key=lambda r: r[0])

    # Flatten rows in top-to-bottom, left-to-right order
    sorted_rects = []
    for row in rows:
        sorted_rects.extend(row)

    # 4. Crop images
    card_images = []
    for rect in sorted_rects[:12]:
        x, y, w, h = rect
        card_img = src[y:y+h, x:x+w]
        card_images.append(card_img)

    return card_images

def process_all_grids(input_folder='grid', output_folder='outputs'):
    # Make sure output folder exists
    os.makedirs(output_folder, exist_ok=True)

    # Grab all .png/.jpg/.jpeg files in the grids folder
    image_extensions = ("*.png", "*.jpg", "*.JPEG")
    image_paths = []
    for ext in image_extensions:
        image_paths.extend(glob(os.path.join(input_folder, ext)))

    # For each image in 'grids', extract cards and save
    for img_path in image_paths:
        cards = extract_cards_3x4(img_path)

        # Base name of the file (e.g. "my_grid" from "my_grid.png")
        base_name = os.path.splitext(os.path.basename(img_path))[0]

        # Save each card in the 'outputs' folder
        for i, card_img in enumerate(cards):
            output_path = os.path.join(output_folder, f"{base_name}_card_{i}.png")
            cv2.imwrite(output_path, card_img)

if __name__ == "__main__":
    process_all_grids()
