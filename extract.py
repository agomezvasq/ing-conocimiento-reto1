import os
import cv2
import numpy as np

avg_dataset = cv2.imread("data/train/avg_dataset.png")

zeros = np.zeros((1080, 1080, 3))

for subdir, dirs, files in os.walk("data/train/cropped"):
    for filename in files:
        if filename.endswith(".jpg"):
            img = cv2.imread(subdir + "/" + filename)

            #cv2.namedWindow("a", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("a", 720, 720)
            #cv2.namedWindow("b", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("b", 720, 720)
            #cv2.namedWindow("c", cv2.WINDOW_NORMAL)
            #cv2.resizeWindow("c", 720, 720)

            #cv2.imshow("a", img)
            #cv2.imshow("b", avg_dataset)

            #cv2.imshow("a", img)

            img2 = cv2.absdiff(img, avg_dataset)

            hsv = cv2.cvtColor(img2, cv2.COLOR_BGR2HSV)

            h, s, v = cv2.split(hsv)

            img2 = np.where(np.logical_and(s > 31, v > 110), 255, 0)

            #img2 = cv2.dilate(img2.astype("uint8"), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=5)

            #img2 = cv2.erode(img2.astype("uint8"), cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=100)

            img2 = cv2.bitwise_and(img, img, mask=img2.astype("uint8"))

            #img = cv2.morphologyEx(img.astype("uint8"), cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)))

            #cv2.imshow("c", img2.astype("uint8"))

            cv2.imwrite("data/train/extracted/" + os.path.basename(subdir) + "/" + filename, img2.astype("uint8"))

            img = cv2.imread("")

            cv2.waitKey(0)

            print("data/train/extracted/" + os.path.basename(subdir) + "/" + filename)