from re import I
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt

import seaborn as sns
import pingouin as pg
import math

from sklearn.preprocessing import StandardScaler




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

    # print("t Student: ")
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

    print("PCA")

    X = df.iloc[:,1:2888].values
    y = df.iloc[:,2888].values

    X_std = StandardScaler().fit_transform(X)
    print('NumPy covariance matrix: \n%s' %np.cov(X_std.T))

    # cov_mat = np.cov(X_std.T)

    # eig_vals, eig_vecs = np.linalg.eig(cov_mat)

    # print('Eigenvectors \n%s' %eig_vecs)
    # print('\nEigenvalues \n%s' %eig_vals)

    # #  Hacemos una lista de parejas (autovector, autovalor) 
    # eig_pairs = [(np.abs(eig_vals[i]), eig_vecs[:,i]) for i in range(len(eig_vals))]

    # # Ordenamos estas parejas den orden descendiente con la función sort
    # eig_pairs.sort(key=lambda x: x[0], reverse=True)

    # # Visualizamos la lista de autovalores en orden desdenciente
    # print('Autovalores en orden descendiente:')
    # for i in eig_pairs:
    #     print(i[0])

    # # A partir de los autovalores, calculamos la varianza explicada
    # tot = sum(eig_vals)
    # var_exp = [(i / tot)*100 for i in sorted(eig_vals, reverse=True)]
    # cum_var_exp = np.cumsum(var_exp)

    # # Representamos en un diagrama de barras la varianza explicada por cada autovalor, y la acumulada
    # with plt.style.context('seaborn-pastel'):
    #     plt.figure(figsize=(6, 4))

    #     plt.bar(range(4), var_exp, alpha=0.5, align='center',
    #             label='Varianza individual explicada', color='g')
    #     plt.step(range(4), cum_var_exp, where='mid', linestyle='--', label='Varianza explicada acumulada')
    #     plt.ylabel('Ratio de Varianza Explicada')
    #     plt.xlabel('Componentes Principales')
    #     plt.legend(loc='best')
    #     plt.tight_layout()

    # #Generamos la matríz a partir de los pares autovalor-autovector
    # matrix_w = np.hstack((eig_pairs[0][1].reshape(4,1),
    #                     eig_pairs[1][1].reshape(4,1)))

    # print('Matriz W:\n', matrix_w)

    # Y = X_std.dot(matrix_w)

    # with plt.style.context('seaborn-whitegrid'):
    #     plt.figure(figsize=(6, 4))
    #     for lab, col in zip(('-1', '0'),
    #                         ('magenta', 'cyan')):
    #         plt.scatter(Y[y==lab, 0],
    #                     Y[y==lab, 1],
    #                     label=lab,
    #                     c=col)
    #     plt.xlabel('Componente Principal 1')
    #     plt.ylabel('Componente Principal 2')
    #     plt.legend(loc='lower center')
    #     plt.tight_layout()
    #     plt.show()
