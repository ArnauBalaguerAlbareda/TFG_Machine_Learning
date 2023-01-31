import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, make_scorer,accuracy_score , recall_score ,f1_score
from sklearn.utils import resample
from .seve_model import *
from sklearn import preprocessing



def K_NN(nameMatrix, graf, meth):

    if(graf == False):
        t_student = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + 't_student.csv')
        PCA_funtion = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_PCA' + ".csv")
    else:
        t_student = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + '_' + str(meth) + "_graph.csv")

    
    ##
    ## t_student
    ##
    x_t = t_student.iloc[:,1:t_student.shape[1]-1]
    y_t = t_student.iloc[:,t_student.shape[1]-1]

    accuracy_t_c = []
    precision_t_c = []
    recall_t_c = []
    f1_t_c = []

    # cross validation -----------------
    lb = preprocessing.LabelBinarizer()
    y_t = lb.fit_transform(y_t).ravel()
    for n in range(1,30):
        Knn = KNeighborsClassifier(n_neighbors=n)
        accuracy_t_c.append(cross_val_score(Knn, x_t, y_t, cv = 7, scoring='accuracy' ).mean())
        precision_t_c.append(cross_val_score(Knn, x_t, y_t, cv = 7, scoring="precision").mean())
        recall_t_c.append(cross_val_score(Knn, x_t, y_t,scoring="recall", cv = 7).mean())
        f1_t_c.append(cross_val_score(Knn, x_t, y_t,scoring="f1", cv = 7).mean())

    pd.DataFrame(accuracy_t_c) .to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_accuracy_t_c.csv')
    pd.DataFrame(precision_t_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_precision_t_c.csv')
    pd.DataFrame(recall_t_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_recall_t_c.csv')
    pd.DataFrame(f1_t_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_f1_t_c.csv')

    plt.plot(range(1,30), accuracy_t_c, label='Accuracy cross')
    plt.plot(range(1,30), precision_t_c, label='Precision cross')
    plt.plot(range(1,30), recall_t_c, label='Recall cross')
    plt.plot(range(1,30), f1_t_c, label='F1 cross')

    plt.legend()
    plt.grid()
    plt.title('K-nn Classifier whit T Student')
    plt.xlabel('neighbor')
    plt.ylabel('%')
    plt.show()

    ##
    ## PCA
    ##
    if(graf == False):

        x_PCA = PCA_funtion.iloc[:,1:PCA_funtion.shape[1]-1]
        y_PCA = PCA_funtion.iloc[:,PCA_funtion.shape[1]-1]

        accuracy_PCA_c = []
        precision_PCA_c = []
        recall_PCA_c = []
        f1_PCA_c = []

        # cross validation -----------------
        lb = preprocessing.LabelBinarizer()
        y_PCA = lb.fit_transform(y_PCA).ravel()
        for n in range(1,30):
            Knn = KNeighborsClassifier(n_neighbors=n)
            accuracy_PCA_c.append(cross_val_score(Knn, x_PCA, y_PCA, cv = 7, scoring='accuracy').mean())
            precision_PCA_c.append(cross_val_score(Knn, x_PCA, y_PCA, cv = 7, scoring="precision").mean())
            recall_PCA_c.append(cross_val_score(Knn, x_PCA, y_PCA, scoring="recall", cv = 7).mean())
            f1_PCA_c.append(cross_val_score(Knn, x_PCA, y_PCA, scoring="f1", cv = 7).mean())


        pd.DataFrame(accuracy_PCA_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_accuracy_PCA_c.csv')
        pd.DataFrame(precision_PCA_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_precision_PCA_c.csv')
        pd.DataFrame(recall_PCA_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_recall_PCA_c.csv')
        pd.DataFrame(f1_PCA_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'knn_f1_PCA_c.csv')

        plt.plot(range(1,30), accuracy_PCA_c, label='Accuracy cross')
        plt.plot(range(1,30), precision_PCA_c, label='Precision cross')
        plt.plot(range(1,30), recall_PCA_c, label='Recall cross')
        plt.plot(range(1,30), f1_PCA_c, label='Recall cross')

        plt.legend()
        plt.grid()
        plt.title('K-nn Classifier whit T Student')
        plt.xlabel('neighbor')
        plt.ylabel('%')
        plt.show()