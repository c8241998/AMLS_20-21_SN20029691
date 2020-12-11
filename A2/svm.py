
from sklearn.metrics import classification_report,accuracy_score
from sklearn import svm



def A2_SVM(training_images, training_labels,val_images, val_labels, test_images, test_labels):
    classifier = svm.SVC(kernel='linear')
    classifier.fit(training_images, training_labels)

    acc_train = accuracy_score(training_labels, classifier.predict(training_images))
    acc_val = accuracy_score(val_labels, classifier.predict(val_images))
    acc_test = accuracy_score(test_labels, classifier.predict(test_images))

    return acc_train,acc_val,acc_test

