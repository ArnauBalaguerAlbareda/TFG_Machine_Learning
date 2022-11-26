from re import I
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns
import pingouin as pg
import math

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def datapreparation(nameMatrix):
    
    df = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + ".csv")
    # df_EM = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_EM' + ".csv" )
    # df_SEM = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_SEM' + ".csv")
    # rang = np.arange(76*76)
    # rang2 = np.arange(76*76+1)

    # print(df)
    
    # print ('Total fields for each entry:', df.shape [ 1 ])
    # print ('Total de entradas:', df.shape [ 0 ])
    # print ('---------------------------------')

    # print('Empty values:', df.isna().sum().sum())
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

    # print ('Standard Deviation:')
    # df_D = pd.DataFrame()
    # df_D["EM"] = list(df_EM.std(numeric_only=True))
    # df_D["SEM"] = list(df_SEM.std(numeric_only=True))
    # diferencia = [e1 - e2 for e1, e2 in zip(df_D["EM"],df_D["SEM"])]
    # df_D["Dif"] = diferencia
    # print(df_D)
    # df_D.to_excel("./data/"+ nameMatrix + "/" + nameMatrix + '_StandardDeviation.xlsx')
    # plt.plot(rang2, df_D["SEM"], 'b-', label='Australia')
    # plt.plot(rang2, df_D["EM"], 'g-', label='New Zealand')
    # plt.show()

    # print("correlation:")
    # correlation = df.corr()
    # correlation['mstype'].to_excel("./data/"+ nameMatrix + "/" + nameMatrix + '_correlation.xlsx')
    # print(correlation['mstype'])

    # winner = df.corr()['mstype'].to_frame().T
    # plt.subplots(figsize=(20, 1))
    # sns.heatmap(winner,center = 0)
    # plt.show()

    # correlation_list=[]
    # for i in range(76): correlation_list.append(0)
    # for i in range(76*76):
    #     if np.isnan(correlation['mstype'][i]) : correlation_list[i//76]+=1

    # print (correlation_list)

    print("Removal of useless columns:")
    df = df.drop(
        df.iloc[:,2888:5777].columns,axis=1
    )
    # df.to_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_1/2' + ".csv")
    print(df)

    # print("---------------------- t Student ----------------------")
    # pval_list = []
    # for i in range(1,2887):
    #     a = df.loc[df.mstype == 0, str(i)]
    #     b = df.loc[df.mstype == -1, str(i)]
    #     res = "{0:.6f}".format(pg.ttest(x=a, y=b, alternative='two-sided', correction=False)['p-val']['T-test'])
    #     if (res == 'nan'): pval_list.append(math.nan)
    #     else: pval_list.append("{0:.6f}".format(pg.ttest(x=a, y=b, alternative='two-sided', correction=False)['p-val']['T-test']))

    # i = np.arange(1,2887)
    # df_pval = pd.DataFrame()
    # df_pval["i"] = i
    # df_pval["pval"] = pval_list
    # df_pval = df_pval[df_pval['pval'].notna()]

    # print(df_pval)
    # df_pval = df_pval.sort_values('pval',ascending=False)
    # print(df_pval)
    # df_pval.to_excel("./data/"+ nameMatrix + "/" + nameMatrix + '_df_pval.xlsx')

    print("---------------------- PCA ----------------------")

    X = df.iloc[:,1:2888].values
    y = df.iloc[:,2888].values
    print(X)
    print(y)
    x_scaled = StandardScaler().fit_transform(X)
    print(x_scaled)
    pca = PCA()
    pca_features = pca.fit_transform(x_scaled)
 
    print('Shape before PCA: ', x_scaled.shape)
    print('Shape after PCA: ', pca_features.shape)
 
    pca_df = pd.DataFrame(
        data=pca_features, 
        columns=np.arange(143))

    target_names = {
        0:'Vs',
        -1:'P'
    }
    
    pca_df['mstype'] = y
    pca_df['mstype'] = pca_df['mstype'].map(target_names)
    
    print(pca_df)