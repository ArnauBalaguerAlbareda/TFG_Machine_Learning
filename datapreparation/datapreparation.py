import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns
import pingouin as pg
import math

from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA


def datapreparation(nameMatrix,graf,meth):
    
    if(graf == False):
        df = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix + ".csv")
        print(df)

        print("---------------------- Remove useless columns ----------------------")
        df = remove_useless_columns(df)
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
        
        print("---------------------- mstype distribution ----------------------")
        sns.countplot(x = 'mstype',data=df)
        df['mstype'].value_counts()
        plt.show()

        print("---------------------- 1,2,3 distribution ----------------------")
        df [[ '1','2','3']] . hist ( figsize = ( 10 , 5 ))
        plt.show()

        print("---------------------- Standard Desviation ----------------------")
        standard_deviation(nameMatrix)

        print("---------------------- Correlation ----------------------")
        correlation(nameMatrix, df)
        
    else: df = pd.read_csv( "./data/"+ nameMatrix + "/" + nameMatrix + '_' + str(meth) + "_graph.csv")

    print("---------------------- t Student ----------------------")
    t_student(nameMatrix, df, graf, meth)

    print("---------------------- PCA ----------------------")
    PCA_funtion(nameMatrix, df, graf, meth)


def standard_deviation(nameMatrix):
    df_EM = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_EM' + ".csv" )
    df_SEM = pd.read_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_SEM' + ".csv")
    df_EM = remove_useless_columns(df_EM)
    df_SEM = remove_useless_columns(df_SEM)
    rang = np.arange(2888)
    df_D = pd.DataFrame()

    df_D["EM"] = list(df_EM.std(numeric_only=True))
    df_D["SEM"] = list(df_SEM.std(numeric_only=True))
    df_D = df_D.drop([0],axis=0)

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



def remove_useless_columns(df):
    df = df.drop(
        df.iloc[:,2889:5777].columns,axis=1
    )
    return df


def t_student(nameMatrix, df, graf, meth):
    pval_list = []
    df_t = pd.DataFrame()
    if(graf == False):range_atributs = 2887 
    else: range_atributs= 76

    for i in range(0,range_atributs):
        a = df.loc[df.mstype == 0, str(i)]
        b = df.loc[df.mstype == -1, str(i)]
        res = "{0:.6f}".format(pg.ttest(x=a, y=b, alternative='two-sided', correction=False )['p-val']['T-test'])
        if (res == 'nan'): pval_list.append(math.nan)
        else: 
            pval_list.append(res) 
            if (float(res)<=0.05): df_t[str(i)] = df[str(i)]

    df_t['mstype']= df['mstype']
    print(df_t)
    df_pval = pd.DataFrame()
    df_pval["pval"] = pval_list
    df_pval["id"] = np.arange(0,range_atributs)
    print(df_pval)
    df_pval = df_pval[df_pval['pval'].notna()]
    df_pval = df_pval.sort_values('pval')
    print(df_pval)
    
    if(graf == False):df_t.to_csv("./data/"+ nameMatrix + "/" + nameMatrix + 't_student.csv')
    else: df_t.to_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_'+ str(meth) + '_t_student.csv')

    # plots t_studen 4 primeros y 4 ultimos
    # print("--- 4r ---")
    # for i in range(0,4):
    #     sns.boxplot(x = df['mstype'],
    #         y = df[str(df_pval.iloc[i, 1])],
    #         hue = df['mstype']
    #     )
    #     plt.show()
    
    # print("--- 4l ---")
    # for i in range(df_pval.shape[0]-4,df_pval.shape[0]):
    #     sns.boxplot(x = df['mstype'],
    #         y = df[str(df_pval.iloc[i, 1])],
    #         hue = df['mstype']
    #     )
    #     plt.show()
        


def PCA_funtion(nameMatrix, df, graf, meth):
    if(graf == False):range_atributs = 2889
    else: range_atributs= 77
    
    x = df.iloc[:,1:range_atributs].values
    y = df.iloc[:,range_atributs].values

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
    if(graf == False):pca_df.to_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_PCA' + ".csv")
    else: pca_df.to_csv("./data/"+ nameMatrix + "/" + nameMatrix +'_'+ str(meth) +'_PCA' + ".csv")