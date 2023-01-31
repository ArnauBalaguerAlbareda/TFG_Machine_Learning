import pandas as pd
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, make_scorer , accuracy_score ,recall_score,f1_score
from sklearn.utils import resample
from .seve_model import *
from sklearn import preprocessing


def decisionTree(nameMatrix , graf, meth):

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

    # cross validation -----------------
    DecisionTree = DecisionTreeClassifier(criterion='entropy')
    lb = preprocessing.LabelBinarizer()
    y_t = lb.fit_transform(y_t)

    accuracy_t_c = (cross_val_score(DecisionTree, x_t, y_t, cv = 7, scoring='accuracy' ).mean())
    precision_t_c = (cross_val_score(DecisionTree, x_t, y_t, cv = 7, scoring="precision").mean())
    recall_t_c = (cross_val_score(DecisionTree, x_t, y_t,scoring="recall", cv = 7).mean())
    f1_t_c = (cross_val_score(DecisionTree, x_t, y_t,scoring="f1", cv = 7).mean())

    ##
    ## PCA
    ##
    if(graf == False):

        x_PCA = PCA_funtion.iloc[:,1:PCA_funtion.shape[1]-1]
        y_PCA = PCA_funtion.iloc[:,PCA_funtion.shape[1]-1]

        # cross validation -----------------
        DecisionTree = DecisionTreeClassifier(criterion='entropy')
        accuracy_PCA_c = (cross_val_score(DecisionTree, x_PCA, y_PCA, cv = 7, scoring='accuracy' ).mean())
        lb = preprocessing.LabelBinarizer()
        y_PCA = lb.fit_transform(y_PCA)
        precision_PCA_c = (cross_val_score(DecisionTree, x_PCA, y_PCA, cv = 7, scoring="precision").mean())
        recall_PCA_c = (cross_val_score(DecisionTree, x_PCA, y_PCA, scoring="recall", cv = 7).mean())
        f1_PCA_c = (cross_val_score(DecisionTree, x_PCA, y_PCA, scoring="f1", cv = 7).mean())