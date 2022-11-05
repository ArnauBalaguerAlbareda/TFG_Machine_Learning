from re import I
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns



def datapreparation(maxmatrix):
    
    df = pd.read_csv('FA' + '.csv')
    # print ('Total de campos para cada entrada:', df.shape [ 1 ])
    # print ('Total de entradas:', df.shape [ 0 ])
    # print ('---------------------------------')

    # print('Valores vacios:', df.isna().sum().sum())
    # print ('---------------------------------')

    # df.info()
    # print ('---------------------------------')

    # print(df.describe())

    # print ('---------------------------------')
    
    # sns.countplot(x = 'mstype',data=df)
    # df['mstype'].value_counts()
    # plt.show()

    # df [[ '1','2','3']] . hist ( figsize = ( 10 , 5 ))
    # plt.show()

    # print ('---------------------------------')
    correlacio = df.corr()
    
    # df.to_csv(maxmatrix + 'datapreparation' + '.csv')
    print(correlacio['mstype'])

    winner = df.corr()['mstype'].to_frame().T 
    plt.subplots(figsize=(20, 1))
    sns.heatmap(winner,center = 0)
    plt.show()








