import cv2
import os
import numpy as np
import tensorflow as tf


def get_features(img):
    r_hist = cv2.calcHist([img], [2], None, [256], [1, 256])
    g_hist = cv2.calcHist([img], [1], None, [256], [1, 256])
    b_hist = cv2.calcHist([img], [0], None, [256], [1, 256])

    r_h = np.array([np.mean(r_hist[:85]), np.mean(r_hist[86:170]), np.mean(r_hist[170:])])
    g_h = np.array([np.mean(g_hist[:85]), np.mean(g_hist[86:170]), np.mean(g_hist[170:])])
    b_h = np.array([np.mean(b_hist[:85]), np.mean(b_hist[86:170]), np.mean(b_hist[170:])])

    r_sigma_h = np.array([np.std(r_hist[:85]), np.std(r_hist[86:170]), np.std(r_hist[170:])])
    g_sigma_h = np.array([np.std(g_hist[:85]), np.std(g_hist[86:170]), np.std(g_hist[170:])])
    b_sigma_h = np.array([np.std(b_hist[:85]), np.std(b_hist[86:170]), np.std(b_hist[170:])])

    return np.concatenate((r_h, g_h, b_h, r_sigma_h, g_sigma_h, b_sigma_h))


classes = ["chocorramo", "flow_blanca", "flow_negra", "frunas_amarilla", "frunas_naranja",
           "frunas_roja", "frunas_verde", "jet_azul", "jumbo_naranja", "jumbo_roja"]
class_dict = {classes[i]: i + 1 for i in range(10)}

f = []

for subdir, dirs, files in os.walk("data/train/object"):
    for filename in files:
        if filename.endswith(".png"):
            img = cv2.imread(subdir + "/" + filename)

            print(subdir + "/" + filename)

            features = get_features(img)

            f.append((features, class_dict[os.path.basename(subdir)]))


feature_names = ["r_h1", "r_h2", "r_h3",
                 "g_h1", "g_h2", "g_h3",
                 "b_h1", "b_h2", "b_h3",
                 "r_sigma_h1", "r_sigma_h2", "r_sigma_h3",
                 "g_sigma_h1", "g_sigma_h2", "g_sigma_h3",
                 "b_sigma_h1", "b_sigma_h2", "b_sigma_h3"]

features = np.array(f).T

print(features.shape)

feature_columns = [tf.feature_column.numeric_column(key=feature_name) for feature_name in feature_names]

