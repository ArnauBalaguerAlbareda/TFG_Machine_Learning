import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import joblib

def seve_model(x,y,bd,t_bd,t_trin,model,name_model,nameMatrix):
    
    if(t_trin == 'c'):
        Knn = model
        joblib.dump(cross_val_score(Knn, x, y, cv = 7),"./data/"+ nameMatrix + "/" + nameMatrix + 'model_' + name_model + '.pkl')
    else:
        
        if(t_bd == 't'):
            t_student_SEM = bd.loc[bd.loc[:,'mstype'] == 0]
            t_student_EM = bd.loc[bd.loc[:,'mstype'] == -1]
            n_size_SEM = int(len(t_student_SEM) * 0.632)
            n_size_EM = int(len(t_student_EM) * 0.632)
        else:
            n_size = int(len(bd) * 0.50)

        Knn = model
        for i in range(7):
            if(t_bd == 't'): 
                train_1 = resample(t_student_SEM.values , n_samples = n_size_SEM)
                train_2 = resample(t_student_EM.values , n_samples = n_size_EM)
                train = np.concatenate((train_1,train_2)) 
            else: 
                train = resample(bd.values , n_samples = n_size)

            test = np.array([x for x in bd.values if x.tolist() not in train.tolist()])
            Knn.fit(train[:,:-1], train[:,-1])

        joblib.dump(Knn,"./data/"+ nameMatrix + "/" + nameMatrix + 'model_' + name_model + '.pkl')
        