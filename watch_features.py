import os
import cv2
import numpy as np
from matplotlib import pyplot as plt
import matplotlib.gridspec as gridspec

WATCH_FEATURES = True


def bounding_box(mask):
    cnts = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    x, y, w, h = 0, 0, 0, 0
    max_area = 0
    for cnt in cnts:
        _x, _y, _w, _h = cv2.boundingRect(cnt)
        area = _w * _h
        if area > max_area:
            max_area = area
            x, y, w, h = _x, _y, _w, _h
    return x, y, w, h


if WATCH_FEATURES:
    SHOW = False
    SAVE = True

    if SHOW:
        windows = ["original", "object_cropped", "masked"]
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

    for subdir, dirs, files in os.walk("data/train/extracted3"):
        for filename in files:
            if filename.endswith(".png"):
                img = cv2.imread(subdir + "/" + filename)
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                _, thresh = cv2.threshold(img, 1, 255, cv2.THRESH_BINARY)

                parts = os.path.splitext(filename)
                original = cv2.imread("data/train/cropped2/" + os.path.basename(subdir) + "/" + parts[0] + ".png")

                masked = cv2.bitwise_and(original, original, mask=img)

                x, y, w, h = bounding_box(thresh)
                #cv2.rectangle(original, (x, y), (x + w, y + h), (0, 255, 0), thickness=4)

                if w == 0 or h == 0:
                    continue

                object_cropped = masked[y:y+h, x:x+w]

                if SAVE:
                    path = "data/train/object/" + os.path.basename(subdir) + "/" + filename
                    cv2.imwrite(path, object_cropped)
                    print(path)

                if SHOW:
                    images = [original, object_cropped, masked]
                    for i in range(len(windows)):
                        cv2.imshow(windows[i], images[i])
                    cv2.waitKey(0)

                    r_hist = cv2.calcHist([object_cropped], [2], None, [256], [1, 256])
                    g_hist = cv2.calcHist([object_cropped], [1], None, [256], [1, 256])
                    b_hist = cv2.calcHist([object_cropped], [0], None, [256], [1, 256])

                    r_h = np.array([np.mean(r_hist[:85]), np.mean(r_hist[86:170]), np.mean(r_hist[170:])])
                    g_h = np.array([np.mean(g_hist[:85]), np.mean(g_hist[86:170]), np.mean(g_hist[170:])])
                    b_h = np.array([np.mean(b_hist[:85]), np.mean(b_hist[86:170]), np.mean(b_hist[170:])])

                    r_sigma_h = np.array([np.std(r_hist[:85]), np.std(r_hist[86:170]), np.std(r_hist[170:])])
                    g_sigma_h = np.array([np.std(g_hist[:85]), np.std(g_hist[86:170]), np.std(g_hist[170:])])
                    b_sigma_h = np.array([np.std(b_hist[:85]), np.std(b_hist[86:170]), np.std(b_hist[170:])])

                    m = max(np.max(r_h), np.max(g_h), np.max(b_h))
                    m_sigma = max(np.max(r_sigma_h), np.max(g_sigma_h), np.max(b_sigma_h))

                    gs = gridspec.GridSpec(3, 4)

                    plt.subplot(gs[:, :2])
                    plt.imshow(cv2.cvtColor(object_cropped, cv2.COLOR_BGR2RGB))
                    ax = plt.subplot(gs[0, 2])
                    ax.set_ylim(0, m)
                    plt.plot(r_h, color='r')
                    ax = plt.subplot(gs[1, 2])
                    ax.set_ylim(0, m)
                    plt.plot(g_h, color='g')
                    ax = plt.subplot(gs[2, 2])
                    ax.set_ylim(0, m)
                    plt.plot(b_h, color='b')
                    ax = plt.subplot(gs[0, 3])
                    ax.set_ylim(0, m_sigma)
                    plt.plot(r_sigma_h, color='r')
                    ax = plt.subplot(gs[1, 3])
                    ax.set_ylim(0, m_sigma)
                    plt.plot(g_h, color='g')
                    ax = plt.subplot(gs[2, 3])
                    ax.set_ylim(0, m_sigma)
                    plt.plot(b_h, color='b')

                    print("r_h: " + str(r_h))
                    print("g_h: " + str(g_h))
                    print("b_h: " + str(b_h))
                    print("r_sigma_h: " + str(r_sigma_h))
                    print("g_sigma_h: " + str(g_sigma_h))
                    print("b_sigma_h: " + str(b_sigma_h))
                    print()

                    plt.show()
                    plt.close()