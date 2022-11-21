from re import I
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns



def datapreparation(maxmatrix):
    
    df = pd.read_csv("./data/"+ maxmatrix + "/" + maxmatrix + ".csv")
    df_EM = pd.read_csv("./data/"+ maxmatrix + "/" + maxmatrix +'_EM' + ".csv" )
    df_SEM = pd.read_csv("./data/"+ maxmatrix + "/" + maxmatrix +'_SEM' + ".csv")
    rang = np.arange(76*76)
    rang2 = np.arange(76*76+1)

    # print(df)
    
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

    print ('Standard Deviation:')
    df_D = pd.DataFrame()
    df_D["EM"] = list(df_EM.std(numeric_only=True))
    df_D["SEM"] = list(df_SEM.std(numeric_only=True))
    diferencia = [e1 - e2 for e1, e2 in zip(df_D["EM"],df_D["SEM"])]
    df_D["Dif"] = diferencia
    print(df_D)
    df_D.to_excel("./data/"+ maxmatrix + "/" + maxmatrix + '_StandardDeviation.xlsx')
    plt.plot(rang2, df_D["SEM"], 'b-', label='Australia')
    plt.plot(rang2, df_D["EM"], 'g-', label='New Zealand')
    plt.show()

    # print("correlation:")
    # correlation = df.corr()
    # correlation['mstype'].to_excel("./data/"+ maxmatrix + "/" + maxmatrix + '_correlation.xlsx')
    # print(correlation['mstype'])

    # winner = df.corr()['mstype'].to_frame().T 
    # plt.subplots(figsize=(20, 1))
    # sns.heatmap(winner,center = 0)
    # plt.show()

    # ll=[]

    # for i in range(76): ll.append(0)
    # for i in range(76*76):
    #     if np.isnan(correlation['mstype'][i]) : ll[i//76]+=1

    # print (ll)