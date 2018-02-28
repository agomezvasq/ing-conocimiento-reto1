import cv2
import numpy as np


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

