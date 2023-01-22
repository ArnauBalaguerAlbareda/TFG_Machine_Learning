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


    # bootstrap ------------------------
    t_student_SEM = t_student.loc[t_student.loc[:,'mstype'] == 0]
    t_student_EM = t_student.loc[t_student.loc[:,'mstype'] == -1]
    n_size_SEM = int(len(t_student_SEM) * 0.632)
    n_size_EM = int(len(t_student_EM) * 0.632)
    accuracy_l = list()
    precision_l = list()
    recall_l = list()
    f1_l = list()

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
        recall_l.append(recall_score(test[:,-1], predictions,pos_label=-1))
        f1_l.append(f1_score(test[:,-1], predictions,pos_label=-1))


    accuracy_t_b = (np.mean(accuracy_l))
    precision_t_b = (np.mean(precision_l))
    recall_t_b = (np.mean(recall_l))
    f1_t_b = (np.mean(f1_l))

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


        # bootstrap ------------------------
        PCA_SEM = PCA_funtion.loc[PCA_funtion.loc[:,'mstype'] == 'HV']
        PCA_EM = PCA_funtion.loc[PCA_funtion.loc[:,'mstype'] == 'MS']
        n_size_SEM = int(len(PCA_SEM) * 0.632)
        n_size_EM = int(len(PCA_EM) * 0.632)
        accuracy_l = list()
        precision_l = list()
        recall_l = list()
        f1_l = list()
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
            recall_l.append(recall_score(test[:,-1], predictions,pos_label='MS'))
            f1_l.append(f1_score(test[:,-1], predictions,pos_label='MS'))

        accuracy_PCA_b = (np.mean(accuracy_l))
        precision_PCA_b = (np.mean(precision_l))
        recall_PCA_b = (np.mean(recall_l))
        f1_PCA_b = (np.mean(f1_l))