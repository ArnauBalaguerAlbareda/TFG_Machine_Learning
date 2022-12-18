import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import RandomizedSearchCV

from sklearn.model_selection import cross_val_score
from .seve_model import *
import multiprocessing
from sklearn.metrics import precision_score


def RN(t_student,PCA_funtion,nameMatrix):

    t_student = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + 't_student.csv')
    PCA_funtion = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_PCA' + ".csv")

    x_t = t_student.iloc[:,1:t_student.shape[1]-1]
    y_t = t_student.iloc[:,t_student.shape[1]-1]

    x_PCA = PCA_funtion.iloc[:,1:PCA_funtion.shape[1]-1]
    y_PCA = PCA_funtion.iloc[:,PCA_funtion.shape[1]-1]

    param_distributions = {
        'hidden_layer_sizes': [(10), (10, 10), (20, 20)],
        'alpha': np.logspace(-3, 3, 7)
    }

    grid_t = RandomizedSearchCV(
        estimator  = MLPClassifier(solver = 'lbfgs', max_iter= 2000),
        param_distributions = param_distributions,
        n_iter     = 50, # Número máximo de combinaciones probadas
        scoring    = 'accuracy',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = 3,
        verbose    = 0,
        random_state = 123,
        return_train_score = True
       )

    grid_t.fit(X = x_t, y = y_t)

    resultados = pd.DataFrame(grid_t.cv_results_)
    print(resultados.filter(regex = '(param.*|mean_t|std_t)')\
        .drop(columns = 'params')\
        .sort_values('mean_test_score', ascending = False)\
        .head(10)
    )

    modelo_t = grid_t.best_estimator_
    print(modelo_t)

    seve_model(x_t,y_t,t_student,'t','c',modelo_t,'RN',nameMatrix)
    seve_model(x_t,y_t,t_student,'t','b',modelo_t,'RN',nameMatrix)


    grid_PCA = RandomizedSearchCV(
        estimator  = MLPClassifier(solver = 'lbfgs', max_iter= 2000),
        param_distributions = param_distributions,
        n_iter     = 50, # Número máximo de combinaciones probadas
        scoring    = 'accuracy',
        n_jobs     = multiprocessing.cpu_count() - 1,
        cv         = 3,
        verbose    = 0,
        random_state = 123,
        return_train_score = True
       )

    grid_PCA.fit(X = x_PCA, y = y_PCA)

    resultados = pd.DataFrame(grid_PCA.cv_results_)
    print(resultados.filter(regex = '(param.*|mean_t|std_t)')\
        .drop(columns = 'params')\
        .sort_values('mean_test_score', ascending = False)\
        .head(10)
    )

    modelo_PCA = grid_PCA.best_estimator_
    print(modelo_PCA)

    seve_model(x_PCA,y_PCA,PCA_funtion,'pca','c',modelo_PCA,'RN',nameMatrix)
    seve_model(x_PCA,y_PCA,PCA_funtion,'pca','b',modelo_PCA,'RN',nameMatrix)