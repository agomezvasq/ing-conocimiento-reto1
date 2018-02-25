import os
import cv2
import numpy as np

MANUAL = False


def colorfulness(img):
    (B, G, R) = cv2.split(img.astype("float"))

    # compute rg = R - G
    rg = np.absolute(R - G)

    # compute yb = 0.5 * (R + G) - B
    yb = np.absolute(0.5 * (R + G) - B)

    # compute the mean and standard deviation of both `rg` and `yb`
    (rbMean, rbStd) = (np.mean(rg), np.std(rg))
    (ybMean, ybStd) = (np.mean(yb), np.std(yb))

    # combine the mean and standard deviations
    stdRoot = np.sqrt((rbStd ** 2) + (ybStd ** 2))
    meanRoot = np.sqrt((rbMean ** 2) + (ybMean ** 2))

    # derive the "colorfulness" metric and return it
    return stdRoot + (0.3 * meanRoot)


if not MANUAL:
    for subdir, dirs, files in os.walk("data/train/videos/frames"):
        for filename in sorted(files, key=lambda x: int(os.path.splitext(x)[0].split('-')[-1])):
            if filename.endswith(".jpg"):
                path = subdir + "/" + filename

                img = cv2.imread(path)

                c = colorfulness(img)
                if c >= 5.5:
                    new_path = "data/train/cropped/" + filename
                else:
                    new_path = "data/train/band/" + filename
                os.rename(path, new_path)
                print(path + " -> " + new_path)

if MANUAL:
    for subdir, dirs, files in os.walk("data/train/videos/frames"):
        for filename in sorted(files, key=lambda x: int(os.path.splitext(x)[0].split('-')[-1])):
            if filename.endswith(".jpg"):
                path = subdir + "/" + filename

                img = cv2.imread(path)

                cv2.namedWindow("image", cv2.WINDOW_NORMAL)
                cv2.resizeWindow("image", 1280, 720)
                cv2.imshow("image", img)

                key = cv2.waitKey(0)
                if key == ord('b'):
                    new_path = "data/train/band/" + filename
                elif key == ord('s'):
                    new_path = "data/train/band/shadows/" + filename
                elif key == ord('d'):
                    new_path = "data/train/deleted/" + filename
                # enter key
                elif key == 13:
                    new_path = "data/train/cropped/" + filename
                os.rename(path, new_path)
            print(path + " -> " + new_path)