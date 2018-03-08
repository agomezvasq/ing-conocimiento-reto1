import cv2
import process
from classify import Type
import features

filename = "data/test/videos/2018-02-23-093504.webm"

cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 1280, 720)

cap = cv2.VideoCapture(filename)

while cap.isOpened():
    ret, img = cap.read()

    t, f, img, masked, (x, y, w, h) = process.process(img)

    if t != Type.BAND:
        cv2.rectangle(masked, (x, y), (x + w, y + h), (0, 0, 255), thickness=2)

        cropped_img = masked[y:y + h, x:x + w]

        print(features.feed(cropped_img))

    cv2.imshow("window", masked)

    cv2.waitKey(33)
