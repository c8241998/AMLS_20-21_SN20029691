import numpy as np
from sklearn import tree
from sklearn.metrics import accuracy_score
from skimage import io
import pandas as pd
from sklearn.ensemble import RandomForestClassifier


def A1_FOREST(training_images, training_labels,val_images, val_labels, test_images, test_labels, n=80):
    classifier = RandomForestClassifier(n_estimators=n)
    classifier.fit(training_images, training_labels)

    acc_train = accuracy_score(training_labels, classifier.predict(training_images))
    acc_val = accuracy_score(val_labels, classifier.predict(val_images))
    acc_test = accuracy_score(test_labels, classifier.predict(test_images))

    return acc_train,acc_val,acc_test



if __name__ == '__main__':
    pass





