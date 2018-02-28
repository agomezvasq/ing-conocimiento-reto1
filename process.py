import os
import cv2
import crop
import classify
from classify import Type
import extract2
import watch_features
import features

TRAIN = False
TEST = False

TRAIN_PATH = "/data/train"
TEST_PATH = "/data/test"


def process(img):
    img = crop.crop(img)
    t = classify.classify(img)
    if t == Type.BAND:
        return t, img, None
    mask = extract2.extract(img)
    img = cv2.bitwise_and(img, img, mask=mask)
    x, y, w, h = watch_features.bounding_box(mask)
    img = img[y:y+h, x:x+w]
    f = features.get_features(img)
    return t, f, img, (x, y, w, h)


def process_all_videos(path):


def process_all_images(path):
    save_path = path + "/object"
    band_path = path + "/band"

    for subdir, dirs, files in os.walk(path + "/raw"):
        for filename in files:
            if filename.endswith(".jpg"):
                img = cv2.imread(subdir + "/" + filename)

                t, f, img, (x, y, w, h) = process(img)
                if t == Type.BAND:
                    os.rename(subdir + "/" + filename, band_path + "/" + filename)
                    continue






if TRAIN:
    process_all_images(TRAIN_PATH)
    process_all_videos(TRAIN_PATH)

if TEST:
    process_all_images(TEST_PATH)
    process_all_videos(TEST_PATH)