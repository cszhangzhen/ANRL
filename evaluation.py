# Evaluation Metric for node classification and link prediction

import numpy as np
import random
import math
import warnings
from sklearn.model_selection import train_test_split
from sklearn.svm import LinearSVC
from sklearn.metrics import *
from sklearn.metrics.pairwise import cosine_similarity


def read_label(inputFileName):
    f = open(inputFileName, "r")
    lines = f.readlines()
    f.close()
    N = len(lines)
    y = np.zeros(N, dtype=int)
    for line in lines:
        l = line.strip("\n\r").split(" ")
        y[int(l[0])] = int(l[1])

    return y


def multiclass_node_classification_eval(X, y, ratio = 0.2, rnd = 2018):
    warnings.filterwarnings("ignore")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size = ratio, random_state = rnd)
    clf = LinearSVC()
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)

    macro_f1 = f1_score(y_test, y_pred, average = "macro")
    micro_f1 = f1_score(y_test, y_pred, average = "micro")

    return macro_f1, micro_f1


def link_prediction_ROC(inputFileName, Embeddings):
    f = open(inputFileName, "r")
    lines = f.readlines()
    f.close()

    X_test = []

    for line in lines:
        l = line.strip("\n\r").split(" ")
        X_test.append([int(l[0]), int(l[1]), int(l[2])])

    y_true = [X_test[i][2] for i in range(len(X_test))]
    y_predict = [cosine_similarity(Embeddings[X_test[i][0], :].reshape(
        1, -1), Embeddings[X_test[i][1], :].reshape(1, -1))[0, 0] for i in range(len(X_test))]
    auc = roc_auc_score(y_true, y_predict)

    if auc < 0.5:
        auc = 1 - auc
        
    return auc


def node_classification_F1(Embeddings, y):
    macro_f1_avg = 0
    micro_f1_avg = 0
    for i in range(10):
        rnd = np.random.randint(2018)
        macro_f1, micro_f1 = multiclass_node_classification_eval(
            Embeddings, y, 0.7, rnd)
        macro_f1_avg += macro_f1
        micro_f1_avg += micro_f1
    macro_f1_avg /= 10
    micro_f1_avg /= 10
    print "Macro_f1 average value: " + str(macro_f1_avg)
    print "Micro_f1 average value: " + str(micro_f1_avg)
