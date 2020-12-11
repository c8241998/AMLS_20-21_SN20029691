from A1.landmarks import train_preprocess,test_preprocess
from A1.forest import A1_FOREST
from A1.knn import A1_KNN
from A1.svm import A1_SVM
from A2.svm import A2_SVM
from B1 import B1_CNN_train,B1_CNN_test
from B2 import B2_CNN_train,B2_CNN_test

def add_acc(acc,task,model,acc_train,acc_val,acc_test):
    if(not acc.__contains__(task)):
        acc[task]={}
    if(not acc[task].__contains__(model)):
        acc[task][model]={}
    acc[task][model]['train'] = acc_train
    acc[task][model]['val'] = acc_val
    acc[task][model]['test'] = acc_test

acc=dict()

# # ======================================================================================================================
# # Data preprocessing for task A

data_train_A= train_preprocess()
print('train set and val set preprocessed for task A')
data_test_A = test_preprocess()
# train 3836  val 959
print('test set preprocessed for task A')

# # ======================================================================================================================
# Task A1
x_train, x_val, y_train, y_val = data_train_A[0]
x_test, y_test = data_test_A[0]
print('-------------A1---------------')

# SVM
acc_train, acc_val, acc_test=A1_SVM(x_train, y_train, x_val, y_val, x_test, y_test)
add_acc(acc,'A1','SVM',acc_train,acc_val,acc_test)
print('A1 SVM finished')

# # KNN
acc_train,acc_val, acc_test=A1_KNN(x_train, y_train, x_val, y_val, x_test, y_test)
add_acc(acc,'A1','KNN',acc_train,acc_val,acc_test)
print('A1 KNN finished')

# FOREST
acc_train,acc_val, acc_test=A1_FOREST(x_train, y_train, x_val, y_val, x_test, y_test)
add_acc(acc,'A1','FOREST',acc_train,acc_val,acc_test)
print('A1 RF finished')


# # ======================================================================================================================
# # Task A2
x_train, x_val, y_train, y_val = data_train_A[1]
x_test, y_test = data_test_A[1]
print('-------------A2---------------')

# SVM
acc_train,acc_val, acc_test=A2_SVM(x_train, y_train, x_val, y_val, x_test, y_test)
add_acc(acc,'A2','SVM',acc_train,acc_val,acc_test)
print('A2 SVM finished')

#
#
# # ======================================================================================================================
# Task B1
print('-------------B1---------------')
acc_train, acc_val = B1_CNN_train.main(root_dir='./B1/')
acc_test = B1_CNN_test.main(root_dir='./B1/')
add_acc(acc,'B1','CNN',acc_train,acc_val,acc_test)
print('B1 CNN finished')


# ======================================================================================================================
# Task B2
print('-------------B2---------------')
acc_train, acc_val = B2_CNN_train.main(root_dir='./B2/')
acc_test = B2_CNN_test.main(root_dir='./B2/')
add_acc(acc,'B2','CNN',acc_train,acc_val,acc_test)
print('B2 CNN finished')
#
# # ======================================================================================================================
import json

print(json.dumps(acc,indent=4,ensure_ascii=False))