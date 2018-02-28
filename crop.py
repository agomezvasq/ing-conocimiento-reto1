import os
import cv2


def crop(img):
    height, width, n_channels = img.shape
    return img[:, int(width / 2 - height / 2):int(width / 2 + height / 2)]


def crop_all_images(path, save_path):
    for filename in os.listdir(path):
        if filename.endswith(".jpg"):
            img = cv2.imread(path + "/" + filename)
            cropped_img = crop(img)
            cv2.imwrite(save_path + "/" + filename, cropped_img)
            print(filename)


def crop_all_videos(path, save_path):
    for filename in os.listdir(path):
        if filename.endswith(".webm"):
            cap = cv2.VideoCapture(path + "/" + filename)

            parts = os.path.splitext(filename)
            os.makedirs(save_path + "/" + parts[0])

            i = 1
            while cap.isOpened():
                ret, img = cap.read()

                if ret:
                    cropped_img = crop(img)
                    cv2.imwrite(save_path + "/" + parts[0] + "/" + parts[0] + "-" + str(i) + ".jpg", cropped_img)
                    i += 1
                else:
                    break
            cap.release()

            print(filename + "(" + str(i) + ") frames")


CROP_IMAGES = False
CROP_VIDEOS = False

if CROP_IMAGES:
    crop_all_images("data/train/raw", "data/train/cropped")

if CROP_VIDEOS:
    crop_all_videos("data/train/videos", "data/train/videos/frames")
