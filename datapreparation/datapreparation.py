import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns
import pingouin as pg
import math

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def datapreparation_matrix(nameMatrix):
    
    df = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + ".csv")
    print(df)
    
    print ('Total fields for each entry:', df.shape [ 1 ])
    print ('Total de entradas:', df.shape [ 0 ])
    print ('---------------------------------')

    print('Empty values:', df.isna().sum().sum())
    print ('---------------------------------')

    df.info()
    print ('---------------------------------')

    print(df.describe())

    print ('---------------------------------')
    
    print("---------------------- mstype distrivution ----------------------")
    sns.countplot(x = 'mstype',data=df)
    df['mstype'].value_counts()
    plt.show()

    print("---------------------- 1,2,3 distrivution ----------------------")
    df [[ '1','2','3']] . hist ( figsize = ( 10 , 5 ))
    plt.show()

    print("---------------------- Standard Deviation ----------------------")
    standard_deviation(nameMatrix)

    print("---------------------- Correlation ----------------------")
    correlation(nameMatrix, df)

    print("---------------------- Remove useless columns ----------------------")
    df = remove_useless_columns(df)
    print(df)

    print("---------------------- t Student ----------------------")
    t_student(nameMatrix, df)

    print("---------------------- PCA ----------------------")
    PCA_funtion(nameMatrix, df)


def standard_deviation(nameMatrix):
    df_EM = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_EM' + ".csv" )
    df_SEM = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_SEM' + ".csv")
    rang = np.arange(76*76+1)
    df_D = pd.DataFrame()

    df_D["EM"] = list(df_EM.std(numeric_only=True))
    df_D["SEM"] = list(df_SEM.std(numeric_only=True))
    diferencia = [e1 - e2 for e1, e2 in zip(df_D["EM"],df_D["SEM"])]
    df_D["Dif"] = diferencia
    df_D.to_excel("./data/"+ nameMatrix + "/" + nameMatrix + '_StandardDeviation.xlsx')

    plt.plot(rang, df_D["SEM"], 'b-', label='Australia')
    plt.plot(rang, df_D["EM"], 'g-', label='New Zealand')
    plt.show()


def correlation(nameMatrix, df):
    correlation = df.corr()
    correlation['mstype'].to_excel("./data/"+ nameMatrix + "/" + nameMatrix + '_correlation.xlsx')
    print(correlation['mstype'])

    winner = df.corr()['mstype'].to_frame().T
    plt.subplots(figsize=(20, 1))
    sns.heatmap(winner,center = 0)
    plt.show()

    correlation_list=[]
    for i in range(76): correlation_list.append(0)
    for i in range(76*76):
        if np.isnan(correlation['mstype'][i]) : correlation_list[i//76]+=1

    print (correlation_list)


def remove_useless_columns(df):
    df = df.drop(
        df.iloc[:,2888:5777].columns,axis=1
    )
    return df


def t_student(nameMatrix, df):
    pval_list = []
    df_t = pd.DataFrame()
    for i in range(0,2887):
        a = df.loc[df.mstype == 0, str(i)]
        b = df.loc[df.mstype == -1, str(i)]
        res = "{0:.6f}".format(pg.ttest(x=a, y=b, alternative='two-sided', correction=False)['p-val']['T-test'])
        if (res == 'nan'): pval_list.append(math.nan)
        else: 
            pval_list.append(res) 
            if (float(res)<=0.05): df_t[str(i)] = df[str(i)]

    df_t['mstype']= df['mstype']
    print(df_t)
    df_pval = pd.DataFrame()
    df_pval["pval"] = pval_list
    df_pval["id"] = np.arange(0,2887)
    print(df_pval)
    df_pval = df_pval[df_pval['pval'].notna()]
    df_pval = df_pval.sort_values('pval')
    print(df_pval)

    df_t.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 't_student.csv')

    print("--- 4r ---")
    for i in range(0,4):
        sns.boxplot(x = df['mstype'],
            y = df[str(df_pval.iloc[i, 1])],
            hue = df['mstype']
        )
        plt.show()
    
    print("--- 4l ---")
    for i in range(df_pval.shape[0]-4,df_pval.shape[0]):
        sns.boxplot(x = df['mstype'],
            y = df[str(df_pval.iloc[i, 1])],
            hue = df['mstype']
        )
        plt.show()
        


def PCA_funtion(nameMatrix, df):
    x = df.iloc[:,1:2888].values
    y = df.iloc[:,2888].values

    pca = PCA().fit(x)
    n_c = np.cumsum(pca.explained_variance_ratio_)
    print(n_c)
    plt.plot(n_c)
    plt.xlabel('number of components')
    plt.ylabel('cumulative explained variance')
    plt.show()

    n_componentes = 0
    for i in range(n_c.shape[0]):
        if n_c[i]>0.95:
            n_componentes = i
            break

    x_scaled = StandardScaler().fit_transform(x)
    pca = PCA(n_componentes)
    pca_features = pca.fit_transform(x_scaled)
 
    print('Shape before PCA: ', x_scaled.shape)
    print('Shape after PCA: ', pca_features.shape)
 
    pca_df = pd.DataFrame(
        data=pca_features, 
        columns=np.arange(n_componentes))

    target_names = {
        0:'HV',
        -1:'MS'
    }
    
    pca_df['mstype'] = y
    pca_df['mstype'] = pca_df['mstype'].map(target_names)
    
    print(pca_df)
    pca_df.to_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_PCA' + ".csv")