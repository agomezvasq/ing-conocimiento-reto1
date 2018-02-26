import os
import cv2
import numpy as np

SHOW = True
SAVE = False

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

for subdir, dirs, files in os.walk("data/train/cropped"):
    for filename in files:
        if filename.endswith(".jpg"):
            img = cv2.imread(subdir + "/" + filename)

            if SHOW:
                windows = ["non-threshold"]
                for window in windows:
                    cv2.namedWindow(window, cv2.WINDOW_NORMAL)
                    cv2.resizeWindow(window, 720, 720)

            fgmask = fgbg.apply(img)

            

            ret, thresh = cv2.threshold(fgmask, 250, 255, 0)

            #fgmask = cv2.morphologyEx(fgmask.astype("uint8"), cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5)), iterations=5)

            masked = cv2.bitwise_and(img, img, mask=fgmask)

            if SAVE:
                cv2.imwrite("data/train/mog/" + os.path.basename(subdir) + "/" + filename, masked)

                print("data/train/mog/" + os.path.basename(subdir) + "/" + filename)

            if SHOW:
                cv2.imshow("non-threshold", fgmask)
                cv2.imshow("threshold", thresh)
                cv2.waitKey(0)