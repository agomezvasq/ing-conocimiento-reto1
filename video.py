import cv2
import process
from classify import Type
import features


def max_key(dct):
    return


filename = "data/test/videos/2018-02-23-093504.webm"

cv2.namedWindow("window", cv2.WINDOW_NORMAL)
cv2.resizeWindow("window", 960, 540)

cap = cv2.VideoCapture(1)

cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)
#cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.3)

classes = {"chocorramo": 0,
           "flow_blanca": 0,
           "flow_negra": 0,
           "frunas_amarilla": 0,
           "frunas_naranja": 0,
           "frunas_roja": 0,
           "frunas_verde": 0,
           "jet_azul": 0,
           "jumbo_naranja": 0,
           "jumbo_roja": 0}

total = {"chocorramo": 0,
         "flow_blanca": 0,
         "flow_negra": 0,
         "frunas_amarilla": 0,
         "frunas_naranja": 0,
         "frunas_roja": 0,
         "frunas_verde": 0,
         "jet_azul": 0,
         "jumbo_naranja": 0,
         "jumbo_roja": 0}

import crop
i = 1

while cap.isOpened():
    ret, img = cap.read()

    #img = cv2.resize(img, (1920, 1080))

    t, f, img, masked, (x, y, w, h) = process.process(img)

    if t != Type.BAND and w != 0 and h != 0:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 255), thickness=2)

        cropped_img = masked[y:y + h, x:x + w]

        prediction = features.feed(cropped_img)

        classes[prediction] += 1

        mx = max(classes, key=classes.get)

        s = sum(classes.values())

        #print(mx + ": " + str(classes[mx]))
        #print(mx)

        cv2.putText(img, prediction, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2, cv2.LINE_AA)

        #print()
    else:
        s = sum(classes.values())

        if s != 0:
            #print({k: str(v / s * 100) + "%" for k, v in classes.items()})

            mx = max(classes, key=classes.get)

            total[mx] += 1

            print(total)

        classes = {"chocorramo": 0,
                   "flow_blanca": 0,
                   "flow_negra": 0,
                   "frunas_amarilla": 0,
                   "frunas_naranja": 0,
                   "frunas_roja": 0,
                   "frunas_verde": 0,
                   "jet_azul": 0,
                   "jumbo_naranja": 0,
                   "jumbo_roja": 0}

    cv2.imshow("window", img)

    cv2.waitKey(33)