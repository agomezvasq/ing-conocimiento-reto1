import cv2
import os
import numpy as np
import tensorflow as tf
import pickle

OVERWRITE = False


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


def shuffle_unison(a, b):
    p = np.random.permutation(len(a))
    return a[p], b[p]


def train_input_fn(features, labels, batch_size):
    dataset = tf.data.Dataset.from_tensor_slices((dict(features), labels))

    return dataset.shuffle(10000).repeat().batch(batch_size)


def eval_input_fn(features, labels, batch_size):
    features = dict(features)
    if labels is None:
        inputs = features
    else:
        inputs = (features, labels)

    dataset = tf.data.Dataset.from_tensor_slices(inputs)

    return dataset.batch(batch_size)


classes = ["chocorramo", "flow_blanca", "flow_negra", "frunas_amarilla", "frunas_naranja",
           "frunas_roja", "frunas_verde", "jet_azul", "jumbo_naranja", "jumbo_roja"]
class_dict = {classes[i]: i for i in range(10)}

feature_names = ["r_h1", "r_h2", "r_h3",
                 "g_h1", "g_h2", "g_h3",
                 "b_h1", "b_h2", "b_h3",
                 "r_sigma_h1", "r_sigma_h2", "r_sigma_h3",
                 "g_sigma_h1", "g_sigma_h2", "g_sigma_h3",
                 "b_sigma_h1", "b_sigma_h2", "b_sigma_h3"]

if not os.path.exists("features/features") or OVERWRITE:
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
                l.append(class_dict[os.path.basename(subdir)])
        if len(files) != 0:
            x, y = shuffle_unison(np.array(f), np.array(l))
            cutoff = int(len(x) * 0.7)
            tr_x.extend(x[:cutoff])
            tr_y.extend(y[:cutoff])
            t_x.extend(x[cutoff:])
            t_y.extend(y[cutoff:])

    train_x = dict(list(zip(feature_names, np.array(tr_x).T)))
    train_y = tr_y
    test_x = dict(list(zip(feature_names, np.array(t_x).T)))
    test_y = t_y

    with open("features/features", "wb") as f:
        pickle.dump((train_x, train_y, test_x, test_y), f)
else:
    with open("features/features", "rb") as f:
        train_x, train_y, test_x, test_y = pickle.load(f)

feature_columns = [tf.feature_column.numeric_column(key=feature_name) for feature_name in feature_names]

classifier = tf.estimator.DNNClassifier(feature_columns=feature_columns,
                                        hidden_units=[36, 36],
                                        n_classes=10,
                                        model_dir="model")

classifier.train(input_fn=lambda: train_input_fn(train_x, train_y, 100),
                 steps=10000)

eval_result = classifier.evaluate(input_fn=lambda: eval_input_fn(test_x, test_y, 100))

print(eval_result)