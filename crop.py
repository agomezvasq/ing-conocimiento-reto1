import os
import cv2

for filename in os.listdir("data/train/raw"):
    if filename.endswith(".jpg"):
        img = cv2.imread("data/train/raw/" + filename)
        height, width, n_channels = img.shape
        cropped_img = img[:, int(width / 2 - height / 2):int(width / 2 + height / 2)]
        cv2.imwrite("data/train/cropped/" + filename, cropped_img)
        print(filename)