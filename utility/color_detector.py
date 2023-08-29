import cv2
import numpy as np

def update_mask(x):
    global lower, upper
    lower[0] = cv2.getTrackbarPos('Hue Lower', 'Mask')
    lower[1] = cv2.getTrackbarPos('Saturation Lower', 'Mask')
    lower[2] = cv2.getTrackbarPos('Value Lower', 'Mask')
    upper[0] = cv2.getTrackbarPos('Hue Upper', 'Mask')
    upper[1] = cv2.getTrackbarPos('Saturation Upper', 'Mask')
    upper[2] = cv2.getTrackbarPos('Value Upper', 'Mask')
    mask = cv2.inRange(hsv_image, lower, upper)
    masked_image = cv2.bitwise_and(image, image, mask=mask)
    cv2.imshow('Masked Image', masked_image)

# Read the input image
image = cv2.imread('image location') #insert you image location
hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)

# Initialize trackbar values for lower and upper HSV ranges
lower = np.array([0, 0, 0])
upper = np.array([179, 255, 255])

# Create a window for the trackbars
cv2.namedWindow('Mask')

# Create trackbars for HSV range
cv2.createTrackbar('Hue Lower', 'Mask', lower[0], 179, update_mask)
cv2.createTrackbar('Saturation Lower', 'Mask', lower[1], 255, update_mask)
cv2.createTrackbar('Value Lower', 'Mask', lower[2], 255, update_mask)
cv2.createTrackbar('Hue Upper', 'Mask', upper[0], 179, update_mask)
cv2.createTrackbar('Saturation Upper', 'Mask', upper[1], 255, update_mask)
cv2.createTrackbar('Value Upper', 'Mask', upper[2], 255, update_mask)

update_mask(0)  # Initialize the mask

cv2.imshow('Original Image', image) #show original image

cv2.waitKey(0)
cv2.destroyAllWindows()