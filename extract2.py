import os
import cv2
import numpy as np

SHOW = False
SAVE = True
SAVE_MASK = True

avg_dataset = cv2.imread("data/train/avg_dataset.png")

fgbg = cv2.createBackgroundSubtractorMOG2(detectShadows=False)

if SHOW:
    windows = ["original", "mog", "morphed_open", "morphed_closed", "masked", "subtracted"]
    i = 0
    j = 0
    size = 500
    for window in windows:
        cv2.namedWindow(window, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(window, size, size)
        cv2.moveWindow(window, i * (size + 100), j * (size + 100))
        i += 1
        if i > 2:
            i = 0
            j += 1

for subdir, dirs, files in os.walk("data/train/cropped"):
    for filename in files:
        if filename.endswith(".jpg"):
            img = cv2.imread(subdir + "/" + filename)

            subtracted = cv2.absdiff(img, avg_dataset)

            fgmask = fgbg.apply(img)

            ret, thresh = cv2.threshold(fgmask, 250, 255, 0)

            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
            morphed_open = cv2.morphologyEx(fgmask.astype("uint8"), cv2.MORPH_OPEN, kernel, iterations=5)
            morphed_closed = cv2.morphologyEx(morphed_open.astype("uint8"), cv2.MORPH_CLOSE, kernel, iterations=20)

            masked = cv2.bitwise_and(img, img, mask=morphed_closed)

            if SAVE:
                if not SAVE_MASK:
                    path = "data/train/mog/" + os.path.basename(subdir) + "/" + filename
                    cv2.imwrite(path, masked)
                    print(path)
                else:
                    parts = os.path.splitext(filename)
                    path = "data/train/mog_mask/" + os.path.basename(subdir) + "/" + parts[0] + ".png"
                    cv2.imwrite(path, morphed_closed)
                    print(path)

            if SHOW:
                images = [img, fgmask, morphed_open, morphed_closed, masked, subtracted]
                for i in range(len(windows)):
                    cv2.imshow(windows[i], images[i])
                cv2.waitKey(0)