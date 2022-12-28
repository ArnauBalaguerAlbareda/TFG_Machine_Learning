import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.svm import SVC
from sklearn.metrics import precision_score, make_scorer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,precision_score
from .seve_model import *

def svm(nameMatrix, graf, meth):
    
    if(graf == False):
        t_student = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + 't_student.csv')
        PCA_funtion = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_PCA' + ".csv")
    else:
        t_student = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_'+ str(meth) + '_t_student.csv')
        PCA_funtion = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_'+ str(meth) +'_PCA' + ".csv")

    x_t = t_student.iloc[:,1:t_student.shape[1]-1]
    y_t = t_student.iloc[:,t_student.shape[1]-1]

    x_PCA = PCA_funtion.iloc[:,1:PCA_funtion.shape[1]-1]
    y_PCA = PCA_funtion.iloc[:,PCA_funtion.shape[1]-1]


    accuracy_t_c = []
    precision_t_c = []
    accuracy_t_b = []
    precision_t_b = []

    # cross validation -----------------
    for n in range(1,30,1):
        svm = SVC(gamma = "auto", kernel = "rbf", C = n)
        accuracy_t_c.append(cross_val_score(svm, x_t, y_t, cv = 7, scoring='accuracy' ).mean())
        precision = make_scorer(precision_score, pos_label=-1)
        precision_t_c.append(cross_val_score(svm, x_t, y_t, cv = 7, scoring=precision).mean())

    # bootstrap ------------------------
    t_student_SEM = t_student.loc[t_student.loc[:,'mstype'] == 0]
    t_student_EM = t_student.loc[t_student.loc[:,'mstype'] == -1]

    n_size_SEM = int(len(t_student_SEM) * 0.632)
    n_size_EM = int(len(t_student_EM) * 0.632)

    for n in range(1,30,1):
        accuracy_l = list()
        precision_l = list()
        svm = SVC(gamma = "auto", kernel = "rbf", C = n)

        for i in range(7):
            train_1 = resample(t_student_SEM.values , n_samples = n_size_SEM)
            train_2 = resample(t_student_EM.values , n_samples = n_size_EM)
            train = np.concatenate((train_1,train_2))
            test = np.array([x for x in t_student.values if x.tolist() not in train.tolist()])
            svm.fit(train[:,:-1], train[:,-1])
            predictions = svm.predict(test[:,:-1])
            accuracy_l.append(accuracy_score(test[:,-1], predictions))
            precision_l.append(precision_score(test[:,-1], predictions,pos_label=-1))
        accuracy_t_b.append(np.mean(accuracy_l))
        precision_t_b.append(np.mean(precision_l))

    pd.DataFrame(accuracy_t_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'svm_accuracy_t_c.csv')
    pd.DataFrame(precision_t_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'svm_precision_t_c.csv')
    pd.DataFrame(accuracy_t_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'svm_accuracy_t_b.csv')
    pd.DataFrame(precision_t_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'svm_precision_t_b.csv')

    plt.plot(range(1,30,1), accuracy_t_c, label='Accuracy cross')
    plt.plot(range(1,30,1), precision_t_c, label='Precision cross')
    plt.plot(range(1,30,1), accuracy_t_b, label='Accuracy Bootstrap ')
    plt.plot(range(1,30,1), precision_t_b, label='Precision Bootstrap')

    plt.legend()
    plt.grid()
    plt.title('SVM Classifier whit T Student')
    plt.xlabel('C')
    plt.ylabel('%')
    plt.show()

    accuracy_PCA_c = []
    precision_PCA_c = []

    accuracy_PCA_b = []
    precision_PCA_b = []

    # cross validation -----------------
    for n in range(1,30,1):
        svm = SVC(gamma = "auto", kernel = "rbf", C = n)
        accuracy_PCA_c.append(cross_val_score(svm, x_PCA, y_PCA, cv = 7, scoring='accuracy').mean())
        precision = make_scorer(precision_score, pos_label='MS')
        precision_PCA_c.append(cross_val_score(svm, x_PCA, y_PCA, cv = 7, scoring=precision).mean())

    # bootstrap ------------------------

    PCA_SEM = PCA_funtion.loc[PCA_funtion.loc[:,'mstype'] == 'HV']
    PCA_EM = PCA_funtion.loc[PCA_funtion.loc[:,'mstype'] == 'MS']

    n_size_SEM = int(len(PCA_SEM) * 0.632)
    n_size_EM = int(len(PCA_EM) * 0.632)

    for n in range(1,30,1):
        accuracy_l = list()
        precision_l = list()
        svm = SVC(gamma = "auto", kernel = "rbf", C = n)

        for i in range(7):
            train_1 = resample(PCA_SEM.values , n_samples = n_size_SEM)
            train_2 = resample(PCA_EM.values , n_samples = n_size_EM)
            train = np.concatenate((train_1,train_2))
            test = np.array([x for x in PCA_funtion.values if x.tolist() not in train.tolist()])
            svm.fit(train[:,:-1], train[:,-1])
            predictions = svm.predict(test[:,:-1])
            accuracy_l.append(accuracy_score(test[:,-1], predictions))
            precision_l.append(precision_score(test[:,-1], predictions,pos_label='MS'))
        accuracy_PCA_b.append(np.mean(accuracy_l))
        precision_PCA_b.append(np.mean(precision_l))

    pd.DataFrame(accuracy_PCA_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'svm_accuracy_PCA_c.csv')
    pd.DataFrame(precision_PCA_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'svm_precision_PCA_c.csv')
    pd.DataFrame(accuracy_PCA_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'svm_accuracy_PCA_b.csv')
    pd.DataFrame(precision_PCA_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'svm_precision_PCA_b.csv')

    plt.plot(range(1,30,1), accuracy_PCA_c, label='Accuracy cross')
    plt.plot(range(1,30,1), precision_PCA_c, label='Precision cross')
    plt.plot(range(1,30,1), accuracy_PCA_b, label='Accuracy Bootstrap')
    plt.plot(range(1,30,1), precision_PCA_b, label='Precision Bootstrap')
    plt.legend()
    plt.grid()
    plt.title('SVM Classifier whit PCA')
    plt.xlabel('C')
    plt.ylabel('%')
    plt.show()