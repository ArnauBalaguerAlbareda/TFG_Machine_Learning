import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score, make_scorer,recall_score,accuracy_score,f1_score
from sklearn.utils import resample
from .seve_model import *
from sklearn import preprocessing



def randomForest(nameMatrix, graf, meth):

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

    accuracy_t_b = []
    precision_t_b = []
    recall_t_b = []
    f1_t_b = []


    # # cross validation -----------------
    lb = preprocessing.LabelBinarizer()
    y_t = lb.fit_transform(y_t).ravel()
    for n in range(100,300,25):
        radomForest = RandomForestClassifier(n_estimators=n, max_features=0.1, random_state=42)
        accuracy_t_c.append(cross_val_score(radomForest, x_t, y_t, cv = 7, scoring='accuracy' ).mean())
        precision_t_c.append(cross_val_score(radomForest, x_t, y_t, cv = 7, scoring="precision").mean())
        recall_t_c.append(cross_val_score(radomForest, x_t, y_t,scoring="recall", cv = 7).mean())
        f1_t_c.append(cross_val_score(radomForest, x_t, y_t,scoring="f1", cv = 7).mean())



    # # bootstrap ------------------------
    t_student_SEM = t_student.loc[t_student.loc[:,'mstype'] == 0]
    t_student_EM = t_student.loc[t_student.loc[:,'mstype'] == -1]
    n_size_SEM = int(len(t_student_SEM) * 0.632)
    n_size_EM = int(len(t_student_EM) * 0.632)    
    for n in range(100,300,25):
        accuracy_l = list()
        precision_l = list()
        recall_l = list()
        f1_l = list()

        radomForest = RandomForestClassifier(n_estimators=n, max_features=0.1, random_state=42)
        for i in range(7):
            train_1 = resample(t_student_SEM.values , n_samples = n_size_SEM)
            train_2 = resample(t_student_EM.values , n_samples = n_size_EM)
            train = np.concatenate((train_1,train_2))
            test = np.array([x for x in t_student.values if x.tolist() not in train.tolist()])
            radomForest.fit(train[:,:-1], train[:,-1])
            predictions = radomForest.predict(test[:,:-1])
            accuracy_l.append(accuracy_score(test[:,-1], predictions))            
            precision_l.append(precision_score(test[:,-1], predictions,pos_label=-1))
            recall_l.append(recall_score(test[:,-1], predictions,pos_label=-1))
            f1_l.append(f1_score(test[:,-1], predictions,pos_label=-1))


        accuracy_t_b.append(np.mean(accuracy_l))
        precision_t_b.append(np.mean(precision_l))
        recall_t_b.append(np.mean(recall_l))
        f1_t_b.append(np.mean(f1_l))


    pd.DataFrame(accuracy_t_c) .to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_accuracy_t_c.csv')
    pd.DataFrame(precision_t_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_precision_t_c.csv')
    pd.DataFrame(recall_t_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_recall_t_c.csv')
    pd.DataFrame(f1_t_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_f1_t_c.csv')

    pd.DataFrame(accuracy_t_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_accuracy_t_b.csv')
    pd.DataFrame(precision_t_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_precision_t_b.csv')
    pd.DataFrame(recall_t_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_recall_t_b.csv')
    pd.DataFrame(f1_t_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_f1_t_b.csv')
    


    plt.plot(range(100,300,25), accuracy_t_c, label='Accuracy cross')
    plt.plot(range(100,300,25), precision_t_c, label='Precision cross')
    plt.plot(range(100,300,25), recall_t_c, label='Recall cross')
    plt.plot(range(100,300,25), f1_t_c, label='f1 cross')

    plt.plot(range(100,300,25), accuracy_t_b, label='Accuracy Bootstrap ')
    plt.plot(range(100,300,25), precision_t_b, label='Precision Bootstrap')
    plt.plot(range(100,300,25), recall_t_b, label='Recall Bootstrap')
    plt.plot(range(100,300,25), f1_t_b, label='f1 Bootstrap')


    plt.legend()
    plt.grid()
    plt.title('randomForest Classifier whit T Student')
    plt.xlabel('n_estimators')
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


        accuracy_PCA_b = []
        precision_PCA_b = []
        recall_PCA_b = []
        f1_PCA_b = []


        # cross validation -----------------
        lb = preprocessing.LabelBinarizer()
        y_PCA = lb.fit_transform(y_PCA).ravel()
        for n in range(100,300,25):
            radomForest = RandomForestClassifier(n_estimators=n, max_features=0.1, random_state=42)
            accuracy_PCA_c.append(cross_val_score(radomForest, x_PCA, y_PCA, cv = 7, scoring='accuracy').mean())
            precision_PCA_c.append(cross_val_score(radomForest, x_PCA, y_PCA, cv = 7, scoring="precision").mean())
            recall_PCA_c.append(cross_val_score(radomForest, x_PCA, y_PCA, scoring="recall", cv = 7).mean())
            f1_PCA_c.append(cross_val_score(radomForest, x_PCA, y_PCA, scoring="f1", cv = 7).mean())


        # bootstrap ------------------------
        PCA_SEM = PCA_funtion.loc[PCA_funtion.loc[:,'mstype'] == 'HV']
        PCA__EM = PCA_funtion.loc[PCA_funtion.loc[:,'mstype'] == 'MS']
        n_size_SEM = int(len(PCA_SEM) * 0.632)
        n_size_EM = int(len(PCA__EM) * 0.632)    
        for n in range(100,300,25):
            accuracy_l = list()
            precision_l = list()
            recall_l = list()
            f1_l = list()
            radomForest = RandomForestClassifier(n_estimators=n, max_features=0.1, random_state=42)
            for i in range(7):
                train_1 = resample(PCA_SEM.values , n_samples = n_size_SEM)
                train_2 = resample(PCA__EM.values , n_samples = n_size_EM)
                train = np.concatenate((train_1,train_2))
                test = np.array([x for x in PCA_funtion.values if x.tolist() not in train.tolist()])
                radomForest.fit(train[:,:-1], train[:,-1])
                predictions = radomForest.predict(test[:,:-1])
                accuracy_l.append(accuracy_score(test[:,-1], predictions))
                precision_l.append(precision_score(test[:,-1], predictions,pos_label='MS'))
                recall_l.append(recall_score(test[:,-1], predictions,pos_label='MS'))
                f1_l.append(f1_score(test[:,-1], predictions,pos_label='MS'))


            accuracy_PCA_b.append(np.mean(accuracy_l))
            precision_PCA_b.append(np.mean(precision_l))
            recall_PCA_b.append(np.mean(recall_l))
            f1_PCA_b.append(np.mean(f1_l))

        pd.DataFrame(accuracy_PCA_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_accuracy_PCA_c.csv')
        pd.DataFrame(precision_PCA_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_precision_PCA_c.csv')
        pd.DataFrame(recall_PCA_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_recall_PCA_c.csv')
        pd.DataFrame(f1_PCA_c).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_f1_PCA_c.csv')

        pd.DataFrame(accuracy_PCA_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_accuracy_PCA_b.csv')
        pd.DataFrame(precision_PCA_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_precision_PCA_b.csv')
        pd.DataFrame(recall_PCA_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_recall_PCA_b.csv')
        pd.DataFrame(f1_PCA_b).to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 'radomForest_f1_PCA_b.csv')

        plt.plot(range(100,300,25), accuracy_PCA_c, label='Accuracy cross')
        plt.plot(range(100,300,25), precision_PCA_c, label='Precision cross')
        plt.plot(range(100,300,25), recall_PCA_c, label='Recall cross')
        plt.plot(range(100,300,25), f1_PCA_c, label='f1 cross')

        plt.plot(range(100,300,25), accuracy_PCA_b, label='Accuracy Bootstrap')
        plt.plot(range(100,300,25), precision_PCA_b, label='Precision Bootstrap')
        plt.plot(range(100,300,25), recall_PCA_b, label='Recall Bootstrap')
        plt.plot(range(100,300,25), f1_PCA_b, label='F1 Bootstrap')


        plt.legend()
        plt.grid()
        plt.title('randomForest Classifier whit PCA')
        plt.xlabel('n_estimators')
        plt.ylabel('%')
        plt.show()
        print("hola")