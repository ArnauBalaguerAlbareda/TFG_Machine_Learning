import pandas as pd
import numpy as np


from sklearn.model_selection import cross_val_score
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import precision_score, make_scorer
from sklearn.utils import resample
from sklearn.metrics import accuracy_score,precision_score

from mlxtend.evaluate import bootstrap_point632_score

def decisionTree(t_student,PCA_funtion,nameMatrix):

    t_student = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + 't_student.csv')
    PCA_funtion = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_PCA' + ".csv")

    x_t = t_student.iloc[:,1:t_student.shape[1]-1]
    y_t = t_student.iloc[:,t_student.shape[1]-1]

    x_PCA = PCA_funtion.iloc[:,1:PCA_funtion.shape[1]-1]
    y_PCA = PCA_funtion.iloc[:,PCA_funtion.shape[1]-1]

    # cross validation -----------------
    Knn = DecisionTreeClassifier(criterion='entropy')
    accuracy_t_c = (cross_val_score(Knn, x_t, y_t, cv = 7, scoring='accuracy' ).mean())
    precision = make_scorer(precision_score, pos_label=-1)
    precision_t_c = (cross_val_score(Knn, x_t, y_t, cv = 7, scoring=precision).mean())
    # accuracy_t_b_ppp = (np.mean((bootstrap_point632_score(Knn, x_t, y_t, method='.632+'))))

    # bootstrap ------------------------
    n_size = int(len(t_student) * 0.632)
    accuracy_l = list()
    precision_l = list()    
    for i in range(7):
        train = resample(t_student.values , n_samples = n_size)
        test = np.array([x for x in t_student.values if x.tolist() not in train.tolist()])
        Knn = DecisionTreeClassifier(criterion='entropy')
        Knn.fit(train[:,:-1], train[:,-1])
        predictions = Knn.predict(test[:,:-1])
        accuracy_l.append(accuracy_score(test[:,-1], predictions))
        precision_l.append(precision_score(test[:,-1], predictions,pos_label=-1))

    accuracy_t_b = (np.mean(accuracy_l))
    precision_t_b = (np.mean(precision_l))

    # accuracy_t_c.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'Decision_accuracy_t_c.csv')
    # precision_t_c.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'Decision_precision_t_c.csv')
    # accuracy_t_b.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'Decision_accuracy_t_b.csv')
    # precision_t_b.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'Decision_precision_t_b.csv')


    Knn = DecisionTreeClassifier(criterion='entropy')
    accuracy_PCA_c = (cross_val_score(Knn, x_PCA, y_PCA, cv = 7, scoring='accuracy' ).mean())
    precision = make_scorer(precision_score, pos_label=-1)
    precision_PCA_c = (cross_val_score(Knn, x_PCA, y_PCA, cv = 7, scoring=precision).mean())

    # bootstrap ------------------------
    n_size = int(len(t_student) * 0.632)
    accuracy_l = list()
    precision_l = list()    
    for i in range(7):
        train = resample(t_student.values , n_samples = n_size)
        test = np.array([x for x in t_student.values if x.tolist() not in train.tolist()])
        Knn = DecisionTreeClassifier(criterion='entropy')
        Knn.fit(train[:,:-1], train[:,-1])
        predictions = Knn.predict(test[:,:-1])
        accuracy_l.append(accuracy_score(test[:,-1], predictions))
        precision_l.append(precision_score(test[:,-1], predictions,pos_label=-1))

    accuracy_PCA_b = (np.mean(accuracy_l))
    precision_PCA_b = (np.mean(precision_l))

    # accuracy_PCA_c.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'Decision_accuracy_t_c.csv')
    # precision_PCA_c.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'Decision_precision_t_c.csv')
    # accuracy_PCA_b.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'Decision_accuracy_t_b.csv')
    # precision_PCA_b.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'Decision_precision_t_b.csv')