import os
import cv2
import numpy as np

_, _, avg_dataset = cv2.split(cv2.cvtColor(cv2.imread("data/train/avg_dataset.png"), cv2.COLOR_BGR2HSV))

zeros = np.zeros(avg_dataset.shape)

#avg_dataset = cv2.merge((zeros, zeros, avg_dataset))

fgbg = cv2.createBackgroundSubtractorMOG2()

for subdir, dirs, files in os.walk("data/train/cropped"):
    for filename in files:
        if filename.endswith(".jpg"):
            img = cv2.imread(subdir + "/" + filename)

            cv2.namedWindow("a", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("a", 720, 720)

            fgmask = fgbg.apply(img)
            #fgmask = cv2.absdiff(fgmask, avg_dataset)

            mask1 = cv2.bitwise_and(img, img, mask=fgmask)

            fgmask = fgbg.apply(img)

            mask2 = cv2.bitwise_and(mask1, mask1, mask=fgmask)

            hsv = cv2.cvtColor(mask2, cv2.COLOR_BGR2HSV)

            h, s, v = cv2.split(hsv)

            mask2 = np.where(s > 31, 255, 0)

            mask3 = cv2.bitwise_and(img, img, mask=mask2.astype("uint8"))

            cv2.imshow("a", mask3)

            cv2.imwrite("data/train/mog/" + os.path.basename(subdir) + "/" + filename, fgmask)

            cv2.waitKey(0)

            print("data/train/mog/" + os.path.basename(subdir) + "/" + filename)