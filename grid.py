import cv2
import numpy as np
import os

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

    # Sort by y first (so topmost cards come first)
    candidate_rects.sort(key=lambda r: r[1])

    # 3. Group rectangles into rows by y-coordinate
    #    Then within each row, sort by x to get left-to-right order.
    rows = []
    current_row = []
    row_threshold = 50  # You may need to tweak this depending on your card spacing

    # We'll keep track of the 'last_y' to decide if a new contour is on a new row
    last_y = None
    for rect in candidate_rects:
        x, y, w, h = rect
        if last_y is None:
            # First rectangle: start the first row
            current_row.append(rect)
            last_y = y
        else:
            # If the difference in y is small enough, consider it the same row
            if abs(y - last_y) < row_threshold:
                current_row.append(rect)
            else:
                # Otherwise, we finished one row, start a new row
                rows.append(current_row)
                current_row = [rect]
            last_y = y

    # Append the last row if not empty
    if current_row:
        rows.append(current_row)

    # Now sort each row by x (left to right)
    for row in rows:
        row.sort(key=lambda r: r[0])

    # Flatten rows in order: top row first, then second row, etc.
    sorted_rects = []
    for row in rows:
        sorted_rects.extend(row)

    # 4. Crop images in the new sorted order
    #    Take only the first 12 if there are more
    card_images = []
    for rect in sorted_rects[:12]:
        x, y, w, h = rect
        card_img = src[y:y+h, x:x+w]
        card_images.append(card_img)

    return card_images

# Usage
output_folder = 'extracted_cards'
os.makedirs(output_folder, exist_ok=True)

cards = extract_cards_3x4('grid_pic.png')
for i, c in enumerate(cards):   
    cv2.imwrite(os.path.join(output_folder, f'card_{i}.png'), c)
