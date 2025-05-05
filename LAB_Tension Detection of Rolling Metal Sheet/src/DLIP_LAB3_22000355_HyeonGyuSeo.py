#**
#******************************************************************************
#* @author  Seo HyeonGyu
#* @Mod	   2525-05-06 by Seo HyeonGyu
#* @brief   DLIP:  LAB - Tension Detection of Rolling Metal Sheet
#*
#******************************************************************************
#*

import numpy as np
import cv2 as cv

def poly_fit(X, Y, n):
    if X.shape[0] != Y.shape[0] or X.ndim != 1 or Y.ndim != 1:
        raise ValueError("X and Y must be 1D arrays with the same length.")

    X = X.astype(np.float64)
    Y = Y.astype(np.float64)

    A = np.zeros((X.shape[0], n + 1), dtype=np.float64)
    for i in range(X.shape[0]):
        x_val = X[i]
        for j in range(n + 1):
            A[i, j] = x_val ** (n - j)

    c, residuals, rank, s = np.linalg.lstsq(A, Y, rcond=None)
    return c


# ============================================== Open the Video File ================================================ #
cap = cv.VideoCapture('LAB3_Video.mp4')
# =================================================== While loop ===================================================== #
while cap.isOpened():
    # Read a single frame from the video
    ret, frame = cap.read()

    # If frame not read successfully (end of video), break the loop
    if not ret:
        break
    # ================================================= Preprocessing ================================================= #
    # Convert the current frame from BGR color space to HSV color space
    hsv = cv.cvtColor(frame, cv.COLOR_BGR2HSV)

    # Split the HSV image into three channels: Hue, Saturation, and Value (Brightness)
    src_h, src_s, src_v = cv.split(hsv)

    # Get the height and width of the V channel (grayscale-like image)
    [src_height, src_width] = src_v.shape

    # Define a trapezoid-shaped Region of Interest (ROI) as a polygon using 4 points
    pts = np.array([[
        [0, 390],  # Upper-left
        [int(src_width / 2 - 150), 390],  # Upper-right
        [int(2 * src_width / 7), src_height],  # Bottom-right
        [0, src_height]  # Bottom-left
    ]], dtype=np.int32)

    # Create a black mask with the same size as the V channel
    mask = np.zeros_like(src_v)
    cv.fillPoly(mask, pts, 255)

    # Apply the mask to the V channel to isolate the ROI area
    roi = cv.bitwise_and(src_v, mask)

    # =========================================== Sobel Edge Detection ================================================ #
    # Apply the Sobel operator to detect edges using second-order derivative
    sobelx = cv.Sobel(roi, cv.CV_64F, 2, 0, ksize=3)
    sobely = cv.Sobel(roi, cv.CV_64F, 0, 2, ksize=3)

    magnitude = cv.magnitude(sobelx, sobely)

    # Clip the magnitude to the 0-255 range and convert to 8-bit unsigned integer for display
    magnitude = np.uint8(np.clip(magnitude, 0, 255))
    # Apply a binary threshold: pixels above 60 become 220 (white), others become 0 (black)
    threshold, sobel_edges = cv.threshold(magnitude, 60, 200, cv.THRESH_BINARY)

    # ================================================ HoughLinesP ===================================================== #
    # Create an empty image to draw the detected lines
    dstP = np.zeros_like(src_v)
    linesP = cv.HoughLinesP(sobel_edges, 1, np.pi / 180, 10, None, 30, 20)

    roi_center_x = int((src_width / 2 - 100) / 2)

    if linesP is not None:
        for i in range(0, len(linesP)):
            # Extract the endpoints of the line segment
            l = linesP[i][0]
            x1, y1, x2, y2 = l

            # Calculate the angle of the line in degrees
            dx = x2 - x1
            dy = y2 - y1
            angle = np.arctan2(dy, dx) * 180 / np.pi

            # ========================= Line Filtering by Position and Angle ========================= #
            if x1 < roi_center_x and x2 < roi_center_x:
                # Filter out near-horizontal and near-vertical lines
                if angle > 5 and abs(angle) < 70:
                    cv.line(dstP, (x1, y1), (x2, y2), (255, 255, 255), 1, cv.LINE_AA)

            # Check if the line is on the right side of ROI
            elif x1 > roi_center_x and x2 > roi_center_x:
                # Filter out undesired angles and retain diagonally descending lines
                if angle < -5 and 30 < abs(angle) < 65:
                    cv.line(dstP, (x1, y1), (x2, y2), (255, 255, 255), 1, cv.LINE_AA)
    # ================================================ Morphology ==================================================== #
    # Create a structuring element of type 'CROSS' with a 3x3 size
    kernel = cv.getStructuringElement(cv.MORPH_CROSS, (3, 3))
    dstP = cv.morphologyEx(dstP, cv.MORPH_ERODE, kernel, iterations=2)
    # ================================================ Poltfitting ====================================================#
    white_pixels = np.argwhere(dstP > 100)

    if white_pixels is not None:
        # Extract the y (row) and x (column) values of the white pixels
        y_values = white_pixels[:, 0]  # Row indices of white pixels
        x_values = white_pixels[:, 1]  # Column indices of white pixels
        # Fit a polynomial (2nd degree) to the white pixels' coordinates
        curve_mask = np.zeros_like(src_v)
        coeff = poly_fit(x_values, y_values, 2)

        curve_x = np.linspace(min(x_values), max(x_values), 100)
        curve_y = coeff[0] * curve_x ** 2 + coeff[1] * curve_x + coeff[2]
        # Combine the x and y values to get the curve points
        curve_points = np.array([curve_x, curve_y], dtype=np.int32).T

        # Draw the fitted curve on the curve_mask
        cv.polylines(curve_mask, [curve_points], isClosed=False, color=255, thickness=1, lineType=cv.LINE_AA)
        cv.polylines(frame, [curve_points], isClosed=False, color=(0, 255, 0), thickness=2, lineType=cv.LINE_AA)

        # ========================================== Calculate Score & Level ==============================================#
        curve_pixels = np.argwhere(curve_mask > 100)
        if curve_pixels is not None :
            y_curve = curve_pixels[:, 0]
            x_curve = curve_pixels[:, 1]

            bottom = np.max(y_curve)
            score = src_height - bottom

            if bottom > src_height - 120:
                level = 1
            elif bottom > src_height - 250:
                level = 2
            else:
                level = 3

            cv.line(frame, (0, src_height - 120), (src_width, src_height - 120), (255, 0, 0), 3, cv.LINE_AA)
            cv.line(frame, (0, src_height - 250), (src_width, src_height - 250), (208, 248, 0), 3, cv.LINE_AA)

            score_buff = f"Score: {score}"
            level_buff = f"Level: {level}"
            cv.putText(frame, score_buff, (int(src_width / 2), 300), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv.putText(frame, level_buff, (int(src_width / 2), 350), cv.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # Display the result for each frame
    cv.imshow('Final Frame', frame)

    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()
