# AMLS_20-21_SN20029691


In this assignment, we should solve four image classification tasks under the background of gender detection, emotion detection,  face shape recognition and eye colour recognition. We design different models for each task. 

In task A1, with the face landmarks as feature input, we design three classifiers including KNN, RF and SVM. In task A2, we only use SVM as the classifier. In task B1 and B2, we design CNN models and add SE-Block specifically for task B2. We design lots of experiments to do hyper-parameters selection, and experiments demonstrate that our models achieve high accuracy in each task.


## Requirements

* Python==3.6
* torch==1.7.0
* scikit-learn==0.23.2
* scikit-image==0.17.2
* keras==2.4.3
* dlib==19.21.0
* numpy==1.18.5
* pandas==1.1.3
* matplotlib==3.3.2

## Code
* The code of models is in every sub-folders of each task: `svm.py` `forest.py` `knn.py` `models.py`
* We extract face landmarks as feature input in task A1 and A2: `landmarks.py`

	Pretrain models: `shape_predictor_68_face_landmarks.dat`
* we train or test our CNN models in task B1 and B2 via: `*_CNN_train.py` `*_CNN_test.py`

	Dataset class: `dataset.py`

	Configuration of CNN training: `config.json`
* Entrance of our code: `main.py`

## Running
	python main.py

## Results

| Task | Model         | Train Acc | Val Acc | Test Acc |
|:----:|---------------|-----------|---------|----------|
|  A1  | KNN           | 0.8657    | 0.8248  | 0.8029   |
|  A1  | Random Forest | 1.0000    | 0.8707  | 0.8483   |
|  A1  | SVM           | 0.9546    | 0.9267  | 0.9143   |
|  A2  | SVM           | 0.9056    | 0.9027  | 0.8999   |
|  B1  | CNN           | 1.0000    | 1.0000  | 0.9996   |
|  B2  | CNN           | 0.9101    | 0.8770  | 0.8508   |
