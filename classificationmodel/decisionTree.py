import pandas as pd
import numpy as np


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, make_scorer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,precision_score
from .seve_model import *

def decisionTree(t_student,PCA_funtion,nameMatrix):

    t_student = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + 't_student.csv')
    PCA_funtion = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_PCA' + ".csv")

    x_t = t_student.iloc[:,1:t_student.shape[1]-1]
    y_t = t_student.iloc[:,t_student.shape[1]-1]

    x_PCA = PCA_funtion.iloc[:,1:PCA_funtion.shape[1]-1]
    y_PCA = PCA_funtion.iloc[:,PCA_funtion.shape[1]-1]

    # cross validation -----------------
    DecisionTree = DecisionTreeClassifier(criterion='entropy')
    accuracy_t_c = (cross_val_score(DecisionTree, x_t, y_t, cv = 7, scoring='accuracy' ).mean())
    precision = make_scorer(precision_score, pos_label=-1)
    precision_t_c = (cross_val_score(DecisionTree, x_t, y_t, cv = 7, scoring=precision).mean())

    # bootstrap ------------------------
    t_student_SEM = t_student.loc[t_student.loc[:,'mstype'] == 0]
    t_student_EM = t_student.loc[t_student.loc[:,'mstype'] == -1]
    n_size_SEM = int(len(t_student_SEM) * 0.632)
    n_size_EM = int(len(t_student_EM) * 0.632)
    accuracy_l = list()
    precision_l = list()
    DecisionTree = DecisionTreeClassifier(criterion='entropy')
 
    for i in range(7):
        train_1 = resample(t_student_SEM.values , n_samples = n_size_SEM)
        train_2 = resample(t_student_EM.values , n_samples = n_size_EM)
        train = np.concatenate((train_1,train_2))
        test = np.array([x for x in t_student.values if x.tolist() not in train.tolist()])
        DecisionTree.fit(train[:,:-1], train[:,-1])
        predictions = DecisionTree.predict(test[:,:-1])
        accuracy_l.append(accuracy_score(test[:,-1], predictions))
        precision_l.append(precision_score(test[:,-1], predictions,pos_label=-1))

    accuracy_t_b = (np.mean(accuracy_l))
    precision_t_b = (np.mean(precision_l))

    DecisionTree = DecisionTreeClassifier(criterion='entropy')
    accuracy_PCA_c = (cross_val_score(DecisionTree, x_PCA, y_PCA, cv = 7, scoring='accuracy' ).mean())
    precision = make_scorer(precision_score, pos_label='MS')
    precision_PCA_c = (cross_val_score(DecisionTree, x_PCA, y_PCA, cv = 7, scoring=precision).mean())

    # bootstrap ------------------------
    PCA_SEM = PCA_funtion.loc[PCA_funtion.loc[:,'mstype'] == 'HV']
    PCA_EM = PCA_funtion.loc[PCA_funtion.loc[:,'mstype'] == 'MS']
    n_size_SEM = int(len(PCA_SEM) * 0.632)
    n_size_EM = int(len(PCA_EM) * 0.632)
    accuracy_l = list()
    precision_l = list()
    DecisionTree = DecisionTreeClassifier(criterion='entropy')
 
    for i in range(7):
        train_1 = resample(PCA_SEM.values , n_samples = n_size_SEM)
        train_2 = resample(PCA_EM.values , n_samples = n_size_EM)
        train = np.concatenate((train_1,train_2))
        test = np.array([x for x in PCA_funtion.values if x.tolist() not in train.tolist()])
        DecisionTree.fit(train[:,:-1], train[:,-1])
        predictions = DecisionTree.predict(test[:,:-1])
        accuracy_l.append(accuracy_score(test[:,-1], predictions))
        precision_l.append(precision_score(test[:,-1], predictions,pos_label='MS'))

    accuracy_PCA_b = (np.mean(accuracy_l))
    precision_PCA_b = (np.mean(precision_l))