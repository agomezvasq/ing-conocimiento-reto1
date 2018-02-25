import cv2
import numpy as np

img = cv2.imread("data/train/avg_band.png")

hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)

h, s, v = cv2.split(hsv)

print(np.max(s))
print(np.mean(s))
print(np.max(v))
print(np.mean(v))