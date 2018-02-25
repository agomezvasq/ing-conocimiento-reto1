import os
import cv2

CROP_IMAGES = False
CROP_VIDEOS = True

if CROP_IMAGES:
    for filename in os.listdir("data/train/raw"):
        if filename.endswith(".jpg"):
            img = cv2.imread("data/train/raw/" + filename)
            height, width, n_channels = img.shape
            cropped_img = img[:, int(width / 2 - height / 2):int(width / 2 + height / 2)]
            cv2.imwrite("data/train/cropped/" + filename, cropped_img)
            print(filename)

if CROP_VIDEOS:
    for filename in os.listdir("data/train/videos"):
        if filename.endswith(".webm"):
            cap = cv2.VideoCapture("data/train/videos/" + filename)

            parts = os.path.splitext(filename)
            os.makedirs("data/train/videos/frames/" + parts[0])

            i = 1
            while cap.isOpened():
                ret, img = cap.read()

                if ret:
                    height, width, n_channels = img.shape
                    cropped_img = img[:, int(width / 2 - height / 2):int(width / 2 + height / 2)]

                    cv2.imwrite("data/train/videos/frames/" + parts[0] + "/" + parts[0] + "-" + str(i) + ".jpg", cropped_img)

                    i += 1
                else:
                    break
            cap.release()

            print(filename + "(" + str(i) + ") frames")