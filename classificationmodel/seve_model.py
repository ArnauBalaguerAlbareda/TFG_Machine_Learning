import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
import joblib

def seve_model(x,y,bd,t_bd,t_trin,model,name_model,nameMatrix):
    
    if(t_trin == 'c'):
        k = 7
        n_samples = len(x)
        fold_size = n_samples // k
        scores = []
        masks = []
        for fold in range(k):
            # Generar una máscara booleana para los datos de test de este fold
            test_mask = np.zeros(n_samples, dtype=bool)
            test_mask[fold * fold_size : (fold + 1) * fold_size] = True
            # Guardar la máscara para visualizarla después
            masks.append(test_mask)
            # Crear los conjuntos de entrenamiento y test utilizando la máscara
            X_test, y_test = x[test_mask], y[test_mask]
            X_train, y_train = x[~test_mask], y[~test_mask]
            # Ajustar el clasificador
            model.fit(X_train, y_train)
            # Obtener el rendimiento y guardarlo
            scores.append(model.score(X_test, y_test))

        joblib.dump(model,"./data/"+ nameMatrix + "/" + nameMatrix + 'model_' + name_model + '.pkl')
    else:
        
        if(t_bd == 't'):
            t_student_SEM = bd.loc[bd.loc[:,'mstype'] == 0]
            t_student_EM = bd.loc[bd.loc[:,'mstype'] == -1]
            n_size_SEM = int(len(t_student_SEM) * 0.632)
            n_size_EM = int(len(t_student_EM) * 0.632)
        else:
            n_size = int(len(bd) * 0.50)

        for i in range(7):
            if(t_bd == 't'): 
                train_1 = resample(t_student_SEM.values , n_samples = n_size_SEM)
                train_2 = resample(t_student_EM.values , n_samples = n_size_EM)
                train = np.concatenate((train_1,train_2)) 
            else: 
                train = resample(bd.values , n_samples = n_size)

            test = np.array([x for x in bd.values if x.tolist() not in train.tolist()])
            model.fit(train[:,:-1], train[:,-1])

        joblib.dump(model,"./data/"+ nameMatrix + "/" + nameMatrix + 'model_' + name_model + '.pkl')
        