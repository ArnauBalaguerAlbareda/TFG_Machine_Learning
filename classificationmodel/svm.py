import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import precision_score, make_scorer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,precision_score
from mlxtend.evaluate import bootstrap_point632_score

def svm(t_student,PCA_funtion,nameMatrix):
    
    t_student = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + 't_student.csv')
    PCA_funtion = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_PCA' + ".csv")

    x_t = t_student.iloc[:,1:t_student.shape[1]-1]
    y_t = t_student.iloc[:,t_student.shape[1]-1]

    x_PCA = PCA_funtion.iloc[:,1:PCA_funtion.shape[1]-1]
    y_PCA = PCA_funtion.iloc[:,PCA_funtion.shape[1]-1]


    accuracy_t_c = []
    precision_t_c = []
    accuracy_t_b_ppp = []
    accuracy_t_b = []
    precision_t_b = []

    # cross validation -----------------
    for n in range(100,1000,100):
        svm = SVC(gamma = "auto", kernel = "rbf", C = n)
        accuracy_t_c.append(cross_val_score(svm, x_t, y_t, cv = 7, scoring='accuracy' ).mean())
        precision = make_scorer(precision_score, pos_label=-1)
        precision_t_c.append(cross_val_score(svm, x_t, y_t, cv = 7, scoring=precision).mean())
        accuracy_t_b_ppp.append(np.mean((bootstrap_point632_score(svm, x_t, y_t, method='.632+'))))

    # bootstrap ------------------------
    n_size = int(len(t_student) * 0.632)
    for n in range(100,1000,100):
        accuracy_l = list()
        precision_l = list()
        for i in range(7):
            train = resample(t_student.values , n_samples = n_size)
            test = np.array([x for x in t_student.values if x.tolist() not in train.tolist()])
            svm = SVC(gamma = "auto", kernel = "rbf", C = n)
            svm.fit(train[:,:-1], train[:,-1])
            predictions = svm.predict(test[:,:-1])
            accuracy_l.append(accuracy_score(test[:,-1], predictions))
            precision_l.append(precision_score(test[:,-1], predictions,pos_label=-1))
        accuracy_t_b.append(np.mean(accuracy_l))
        precision_t_b.append(np.mean(precision_l))

    accuracy_t_c.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_accuracy_t_c.csv')
    precision_t_c.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_precision_t_c.csv')
    accuracy_t_b.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_accuracy_t_b.csv')
    precision_t_b.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_precision_t_b.csv')

    plt.plot(range(1,30), accuracy_t_c, label='Accuracy cross')
    plt.plot(range(1,30), precision_t_c, label='Precision cross')
    plt.plot(range(1,30), accuracy_t_b, label='Accuracy Bootstrap ')
    plt.plot(range(1,30), precision_t_b, label='Precision Bootstrap')
    plt.plot(range(1,30), accuracy_t_b_ppp, label='Accuracy')

    plt.legend()
    plt.grid()
    plt.title('K-nn Classifier whit T Student')
    plt.xlabel('neighbor')
    plt.ylabel('%')
    plt.show()

    accuracy_PCA_c = []
    precision_PCA_c = []

    accuracy_PCA_b = []
    precision_PCA_b = []

    # cross validation -----------------
    for n in range(100,1000,100):
        svm = SVC(gamma = "auto", kernel = "rbf", C = n)
        accuracy_PCA_c.append(cross_val_score(svm, x_PCA, y_PCA, cv = 5, scoring='accuracy').mean())
        precision = make_scorer(precision_score, pos_label='MS')
        precision_PCA_c.append(cross_val_score(svm, x_PCA, y_PCA, cv = 5, scoring=precision).mean())

    # bootstrap ------------------------
    n_size = int(len(PCA_funtion) * 0.50)
    for n in range(100,1000,100):
        accuracy_l = list()
        precision_l = list()
        for i in range(7):
            train = resample(PCA_funtion.values , n_samples = n_size)
            test = np.array([x for x in PCA_funtion.values if x.tolist() not in train.tolist()])
            svm = SVC(gamma = "auto", kernel = "rbf", C = n)
            svm.fit(train[:,:-1], train[:,-1])
            predictions = svm.predict(test[:,:-1])
            accuracy_l.append(accuracy_score(test[:,-1], predictions))
            precision_l.append(precision_score(test[:,-1], predictions,pos_label='MS'))
        accuracy_PCA_b.append(np.mean(accuracy_l))
        precision_PCA_b.append(np.mean(precision_l))

    accuracy_PCA_c.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_accuracy_PCA_c.csv')
    precision_PCA_c.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_precision_PCA_c.csv')
    accuracy_PCA_b.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_accuracy_PCA_b.csv')
    precision_PCA_b.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_precision_PCA_b.csv')

    plt.plot(range(1,30), accuracy_PCA_c, label='Accuracy cross')
    plt.plot(range(1,30), precision_PCA_c, label='Precision cross')
    plt.plot(range(1,30), accuracy_PCA_b, label='Accuracy Bootstrap')
    plt.plot(range(1,30), precision_PCA_b, label='Precision Bootstrap')
    plt.legend()
    plt.grid()
    plt.title('K-nn Classifier whit T Student')
    plt.xlabel('neighbor')
    plt.ylabel('%')
    plt.show()
