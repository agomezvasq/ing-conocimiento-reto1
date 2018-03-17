import os
import cv2
import numpy as np
import random
import crop

n = 150

avg = np.zeros((1080, 720))

i = 0

lst = list(os.listdir("data/train/band"))
random.shuffle(lst)

selection = lst[:n]

for filename in selection:
    img = cv2.imread("data/train/band/" + filename, cv2.IMREAD_GRAYSCALE)

    height, width = img.shape

    if height == 1080:
        avg += img.astype("float") / n
        i += 1
        print(i)

#avg = crop.crop(avg)
cv2.imwrite("data/train/avg_random_band.png", avg)