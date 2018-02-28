import os
import cv2
import numpy as np

SHOW = False

avg_dataset = cv2.imread("data/train/avg_dataset.png")

for subdir, dirs, files in os.walk("data/train/cropped"):
    for filename in files:
        if filename.endswith(".jpg"):
            img = cv2.imread(subdir + "/" + filename)
            img2 = cv2.absdiff(img, avg_dataset)
            hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)
            h, s, v = cv2.split(hsv)
            img2 = np.where(np.logical_and(s > 31, v > 110), 255, 0)
            img2 = cv2.bitwise_and(img, img, mask=img2.astype("uint8"))
            cv2.imwrite("data/train/extracted/" + os.path.basename(subdir) + "/" + filename, img2.astype("uint8"))
            print("data/train/extracted/" + os.path.basename(subdir) + "/" + filename)