import os
import cv2
import numpy as np

n = 4493
print(n)

avg = np.zeros((1080, 1080, 3))

i = 0

for subdir, dirs, files in os.walk("data/train/cropped"):
    for filename in files:
        if filename.endswith(".jpg"):
            img = cv2.imread(subdir + "/" + filename)

            height, width, channels = img.shape

            if height == 1080:
                avg += img.astype("float") / n
                i += 1
                print(i)
            else:
                print(subdir + "/" + filename)

cv2.imwrite("data/train/avg_dataset.png", avg)