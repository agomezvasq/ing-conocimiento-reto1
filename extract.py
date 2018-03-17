import os
import cv2
import numpy as np

EXTRACT = False


def extract(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    gray = cv2.equalizeHist(gray)
    # clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    # gray = clahe.apply(gray)

    subtracted = cv2.absdiff(avg_band, gray)

    # subtracted = cv2.equalizeHist(subtracted)

    _, mask = cv2.threshold(subtracted, 110, 255, cv2.THRESH_BINARY)

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

    morphed_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=3)

    morphed_closed = cv2.morphologyEx(morphed_open, cv2.MORPH_CLOSE, kernel, iterations=6)

    return morphed_closed, mask, morphed_open


def normalized_rgb(img):
    b, g, r = cv2.split(img.astype("float"))

    n_rgb = np.zeros(img.shape)

    sm = b + g + r

    n_rgb[:, :, 0] = b / sm
    n_rgb[:, :, 1] = g / sm
    n_rgb[:, :, 2] = r / sm

    return (n_rgb * 255.0).astype("uint8")


avg_band = cv2.imread("data/train/avg_random_band.png")

#avg_band = normalized_rgb(avg_band)

avg_band = cv2.cvtColor(avg_band, cv2.COLOR_BGR2GRAY)

#avg_band = cv2.equalizeHist(avg_band)


classes = {"chocorramo": {"threshold": 118, "open": 3, "close": 6},
           "flow_blanca": {"threshold": 118, "open": 3, "close": 6},
           "flow_negra": {"threshold": 118, "open": 3, "close": 6},
           "frunas_amarilla": {"threshold": 118, "open": 3, "close": 6},
           "frunas_naranja": {"threshold": 118, "open": 3, "close": 6},
           "frunas_roja": {"threshold": 120, "open": 1, "close": 6},
           "frunas_verde": {"threshold": 118, "open": 3, "close": 6},
           "jet_azul": {"threshold": 118, "open": 3, "close": 6},
           "jumbo_naranja": {"threshold": 118, "open": 3, "close": 6},
           "jumbo_roja": {"threshold": 118, "open": 3, "close": 6}}


if EXTRACT:
    SHOW = False
    SAVE = True
    SAVE_MASK = False

    if SHOW:
        windows = ["original", "masked1", "gray", "subtracted", "mask", "morphed_open", "morphed_closed", "masked"]
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

    for subdir, dirs, files in os.walk("data/train/cropped2"):
        basename = os.path.basename(subdir)
        #if basename == "chocorramo":
        #threshold = classes[basename]["threshold"]
        #open_i = classes[basename]["open"]
        #close_i = classes[basename]["close"]

        threshold = 110
        open_i = 3
        close_i = 6

        for filename in files:
            if filename.endswith(".png"):
                img = cv2.imread(subdir + "/" + filename)

                #n_rgb = normalized_rgb(img)

                #b, g, r = cv2.split(img)

                #b = cv2.equalizeHist(b)
                #g = cv2.equalizeHist(g)
                #r = cv2.equalizeHist(r)

                #img2 = cv2.merge((b, g, r))

                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

                gray = cv2.equalizeHist(gray)
                #clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                #gray = clahe.apply(gray)


                subtracted = cv2.absdiff(avg_band, gray)

                #subtracted = cv2.equalizeHist(subtracted)

                _, mask = cv2.threshold(subtracted, threshold, 255, cv2.THRESH_BINARY)

                kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))

                morphed_open = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=open_i)

                morphed_closed = cv2.morphologyEx(morphed_open, cv2.MORPH_CLOSE, kernel, iterations=close_i)

                masked = cv2.bitwise_and(img, img, mask=morphed_closed.astype("uint8"))

                if SAVE:
                    path = "data/train/extracted3/" + os.path.basename(subdir) + "/" + filename
                    cv2.imwrite(path, masked.astype("uint8"))
                    print(path)

                if SHOW:
                    images = [img, masked, gray, subtracted, mask, morphed_open, morphed_closed, masked]
                    for i in range(len(windows)):
                        cv2.imshow(windows[i], images[i])
                    cv2.waitKey(0)