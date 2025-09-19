import numpy as np
import cv2

img = np.zeros([224, 224, 3], np.uint8)
img.fill(255)

mask = np.zeros(img.shape[:2], np.uint8)

cv2.rectangle(mask, (56, 56), (84, 84), 255, -1)

mask_inv = cv2.bitwise_not(mask)
masked_topleft = cv2.bitwise_and(img, img, mask = mask_inv)