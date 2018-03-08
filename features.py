import cv2
import os
import numpy as np
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.models import load_model
import pickle

OVERWRITE = False
TRAIN = True


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

    img = cv2.copyMakeBorder(img, 10, 10, 10, 10, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    _, thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY)

    cnts = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[1]
    max_area = 0
    max_area_cnt = None
    for cnt in cnts:
        _x, _y, _w, _h = cv2.boundingRect(cnt)
        area = _w * _h
        if area > max_area:
            max_area = area
            max_area_cnt = cnt

    #cont = thresh.copy()
    #cv2.drawContours(cont, cnts, -1, (0, 0, 255), thickness=2)
    #cv2.imshow("window", cont)
    #cv2.waitKey(0)

    #hull = cv2.convexHull(max_area_cnt)

    #print(max_area)

    _, (width, height), _ = cv2.minAreaRect(max_area_cnt)

    area = width * height

    #r_h /= 10000
    #g_h /= 10000
    #b_h /= 10000

    #r_sigma_h /= 16384
    #g_sigma_h /= 16384
    #b_sigma_h /= 16384

    #mx = np.sqrt(720 ** 2 + 1080 ** 2)

    #width /= mx
    #height /= mx
    #area /= 720 * 1080

    #cont = img.copy()
    #box = cv2.boxPoints(rect)
    #box = np.int0(box)
    #cv2.drawContours(cont, [box], 0, (255, 255, 255), thickness=2)
    #cv2.imshow("window", cont)
    #cv2.waitKey(0)

    #print(rect)

    #print(width, height, area)

    #print(r_h, g_h, b_h, r_sigma_h, g_sigma_h, b_sigma_h, np.array([width, height, area]))

    return np.concatenate((r_h, g_h, b_h, r_sigma_h, g_sigma_h, b_sigma_h, np.array([width, height, area])))


def shuffle_unison(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


def normalize(a):
    a_prime = (a - np.mean(a, 0)) / np.std(a, 0)

    return (a_prime + 1) / 2


def feed(img):
    features = get_features(img)

    x = np.array([features])

    x = (x - mean) / std

    prediction = model.predict(x, batch_size=10)[0]

    index = np.argmax(prediction)

    return classes[index], "Probability: " + "{0:.2f}".format(prediction[index] * 100) + "%"


classes = ["chocorramo", "flow_blanca", "flow_negra", "frunas_amarilla", "frunas_naranja",
           "frunas_roja", "frunas_verde", "jet_azul", "jumbo_naranja", "jumbo_roja"]
class_dict = {classes[i]: i for i in range(10)}

if not os.path.exists("a/features") or OVERWRITE:
    tr_x = []
    tr_y = []
    t_x = []
    t_y = []

    for subdir, dirs, files in os.walk("data/train/object"):
        f = []
        l = []
        for filename in files:
            if filename.endswith(".png"):
                img = cv2.imread(subdir + "/" + filename)

                print(subdir + "/" + filename)

                features = get_features(img)

                f.append(features)
                label = class_dict[os.path.basename(subdir)]
                arr = np.zeros(10)
                arr[label] = 1
                l.append(arr)
        if len(files) != 0:
            x, y = shuffle_unison(np.array(f), np.array(l))
            cutoff = int(len(x) * 0.7)
            tr_x.extend(x[:cutoff])
            tr_y.extend(y[:cutoff])
            t_x.extend(x[cutoff:])
            t_y.extend(y[cutoff:])

    train_x = np.array(tr_x)
    train_y = np.array(tr_y)
    test_x = np.array(t_x)
    test_y = np.array(t_y)

    with open("a/features", "wb") as f:
        pickle.dump((train_x, train_y, test_x, test_y), f)
else:
    with open("a/features", "rb") as f:
        train_x, train_y, test_x, test_y = pickle.load(f)

        #print(np.mean(train_x))
        #print(np.std(train_x))
        #print(np.mean(test_x))
        #print(np.std(test_x))

mean = np.mean(train_x)
std = np.std(train_x)

train_x = (train_x - mean) / std
test_x = (test_x - mean) / std

mn = np.min(train_x)
mx = np.max(train_x)

#train_x = (train_x + 1) / 2
#test_x = (test_x + 1) / 2

print(mean)
print(std)

print(np.mean(train_x))
print(np.std(train_x))
print(np.mean(test_x))
print(np.std(test_x))

print(np.min(train_x))
print(np.max(train_x))
print(np.min(test_x))
print(np.max(test_x))

if not os.path.exists("model.h5"):
    model = Sequential()

    model.add(Dense(units=36, activation='relu', input_dim=21))
    model.add(Dense(units=10, activation='softmax'))

    sgd = SGD(decay=0.01)
    model.compile(loss='categorical_crossentropy',
                  optimizer=sgd,
                  metrics=['accuracy'])

    if TRAIN:
        model.fit(train_x, train_y, epochs=3000, batch_size=100)

        model.save('model.h5')

    loss_and_metrics = model.evaluate(test_x, test_y, batch_size=100)

    print(loss_and_metrics)
else:
    model = load_model("model.h5")

