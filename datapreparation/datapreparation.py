from re import I
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
import seaborn as sns



def datapreparation(maxmatrix):
    
    df = pd.read_csv(maxmatrix + '.csv')
    df_EM = pd.read_csv(maxmatrix +'_EM' + '.csv')
    df_SEM = pd.read_csv(maxmatrix +'_SEM' + '.csv')

    print ('Total de campos para cada entrada:', df.shape [ 1 ])
    print ('Total de entradas:', df.shape [ 0 ])
    print ('---------------------------------')

    print('Valores vacios:', df.isna().sum().sum())
    print ('---------------------------------')

    df.info()
    print ('---------------------------------')

    print(df.describe())

    print ('---------------------------------')
    
    sns.countplot(x = 'mstype',data=df)
    df['mstype'].value_counts()
    plt.show()

    df [[ '1','2','3']] . hist ( figsize = ( 10 , 5 ))
    plt.show()

    print ('Standard Deviation:')
    df_D = pd.DataFrame()
    df_D["EM"] = list(df_EM.std(numeric_only=True))
    df_D["SEM"] = list(df_SEM.std(numeric_only=True))
    print(df_D)
    df_D.to_excel(maxmatrix + '_StandardDeviation.xlsx')

    print("correlation:")
    correlation = df.corr()
    correlation['mstype'].to_excel(maxmatrix + '_correlation.xlsx')
    print(correlation['mstype'])

    winner = df.corr()['mstype'].to_frame().T 
    plt.subplots(figsize=(20, 1))
    sns.heatmap(winner,center = 0)
    plt.show()

    ll=[]

    for i in range(76): ll.append(0)
    for i in range(76*76):
        if np.isnan(correlation['mstype'][i]) : ll[i//76]+=1

    print (ll)