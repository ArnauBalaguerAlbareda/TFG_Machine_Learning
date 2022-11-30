import pandas as pd
from matplotlib import pyplot as plt
import numpy as np


from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import precision_score, make_scorer

from mlxtend.evaluate import bootstrap_point632_score


def K_NN(t_student,PCA_funtion,nameMatrix):

    t_student = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + 't_student.csv')
    PCA_funtion = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_PCA' + ".csv")

    x_t = t_student.iloc[:,1:t_student.shape[1]-1]
    y_t = t_student.iloc[:,t_student.shape[1]-1]

    x_PCA = PCA_funtion.iloc[:,1:PCA_funtion.shape[1]-1]
    y_PCA = PCA_funtion.iloc[:,PCA_funtion.shape[1]-1]


    accuracy_t_c = []
    precision_t_c = []
    
    accuracy_t_b = []
    precision_t_b = []


    for n in range(1,30):
        Knn = KNeighborsClassifier(n_neighbors=n)
        accuracy_t_c.append(cross_val_score(Knn, x_t, y_t, cv = 5, scoring='accuracy' ).mean())
        precision = make_scorer(precision_score, pos_label=-1)
        precision_t_c.append(cross_val_score(Knn, x_t, y_t, cv = 5, scoring=precision).mean())
        accuracy_t_b.append(np.mean((bootstrap_point632_score(Knn, x_t, y_t, method='.632+',scoring_func=precision))))



    plt.plot(range(1,30), accuracy_t_c, label='Accuracy')
    plt.plot(range(1,30), precision_t_c, label='Precision')
    plt.legend()
    plt.grid()
    plt.title('K-nn Classifier whit T Student')
    plt.xlabel('neighbor')
    plt.ylabel('%')
    plt.show()

    # accuracy_PCA = []
    # precision_PCA = []

    # for n in range(1,30):
    #     Knn = KNeighborsClassifier(n_neighbors=n)
    #     accuracy_PCA.append(cross_val_score(Knn, x_PCA, y_PCA, cv = 5, scoring='accuracy').mean())
    #     precision = make_scorer(precision_score, pos_label='MS')
    #     precision_PCA.append(cross_val_score(Knn, x_PCA, y_PCA, cv = 5, scoring=precision).mean())

    # plt.plot(range(1,30), accuracy_PCA, label='Accuracy')
    # plt.plot(range(1,30), precision_PCA, label='Precision')
    # plt.legend()
    # plt.grid()
    # plt.title('K-nn Classifier whit T Student')
    # plt.xlabel('neighbor')
    # plt.ylabel('%')
    # plt.show()


