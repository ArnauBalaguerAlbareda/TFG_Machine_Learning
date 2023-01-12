import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.metrics import precision_score,accuracy_score, recall_score ,f1_score

def seve_model( x, y, bd, t_bd, t_trin, model, name_model, nameMatrix, graf, meth):

    if(t_bd == 't'):
        label = -1
    else:
        label = 'MS'

    if(t_trin == 'c'):
        k = 7
        n_samples = len(x)
        fold_size = n_samples // k
        scores_a = []
        scores_p = []
        scores_r = []
        scores_f1 = []


        masks = []
        for fold in range(k):
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[fold * fold_size : (fold + 1) * fold_size] = True
            masks.append(test_mask)
            X_test, y_test = x[test_mask], y[test_mask]
            X_train, y_train = x[~test_mask], y[~test_mask]
            model.fit(X_train, y_train)
            predictions = model.predict(X_test)
            scores_a.append(model.score(X_test, y_test))
            scores_p.append(precision_score(y_test, predictions,pos_label=label))
            scores_r.append(recall_score(y_test, predictions,pos_label=label))
            scores_f1.append(f1_score(y_test, predictions,pos_label=label))

        accuracy = np.mean(scores_a)
        precision = np.mean(scores_p)
        recall = np.mean(scores_r)
        f1 = np.mean(scores_f1)


        # if(graf == False):joblib.dump(model,"./data/"+ nameMatrix + "/" + nameMatrix + 'model_' + name_model + '.pkl')
        # else: joblib.dump(model,"./data/"+ nameMatrix + "/" + nameMatrix + 'model_' + name_model + "_" + str(meth) + '.pkl')
    else:
        accuracy_l = list()
        precision_l = list()
        recall_l = list()
        f1_l = list()

        if(t_bd == 't'):
            t_student_SEM = bd.loc[bd.loc[:,'mstype'] == 0]
            t_student_EM = bd.loc[bd.loc[:,'mstype'] == -1]
            n_size_SEM = int(len(t_student_SEM) * 0.632)
            n_size_EM = int(len(t_student_EM) * 0.632)
        else:
            PCA_SEM = bd.loc[bd.loc[:,'mstype'] == 'HV']
            PCA_EM = bd.loc[bd.loc[:,'mstype'] == 'MS']
            n_size_SEM = int(len(PCA_SEM) * 0.632)
            n_size_EM = int(len(PCA_EM) * 0.632)

        for i in range(7):
            if(t_bd == 't'): 
                train_1 = resample(t_student_SEM.values , n_samples = n_size_SEM)
                train_2 = resample(t_student_EM.values , n_samples = n_size_EM)
                train = np.concatenate((train_1,train_2))
            else: 
                train_1 = resample(PCA_SEM.values , n_samples = n_size_SEM)
                train_2 = resample(PCA_EM.values , n_samples = n_size_EM)
                train = np.concatenate((train_1,train_2))

            test = np.array([x for x in bd.values if x.tolist() not in train.tolist()])
            model.fit(train[:,:-1], train[:,-1])
            predictions = model.predict(test[:,:-1])
            accuracy_l.append(accuracy_score(test[:,-1], predictions))
            precision_l.append(precision_score(test[:,-1], predictions,pos_label=label))
            recall_l.append(recall_score(test[:,-1], predictions,pos_label=label))
            f1_l.append(recall_score(test[:,-1], predictions,pos_label=label))

        accuracy = np.mean(accuracy_l)
        precision = np.mean(precision_l)
        recall = np.mean(recall_l)
        f1 = np.mean(f1_l)


        # if(graf == False):joblib.dump(model,"./data/"+ nameMatrix + "/" + nameMatrix + 'model_' + name_model + '.pkl')
        # else: joblib.dump(model,"./data/"+ nameMatrix + "/" + nameMatrix + 'model_' + name_model + "_" + str(meth) + '.pkl')