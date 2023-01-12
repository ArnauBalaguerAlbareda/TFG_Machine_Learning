from re import I
import pandas as pd
import numpy as np
import networkx as nx



def configuration_matrix(data):
    
    rang = np.arange(76*76)
    df = pd.DataFrame(columns=rang)
    df_EM = pd.DataFrame(columns=rang)
    df_SEM = pd.DataFrame(columns=rang)

    res = pd.read_csv("./data/demographics.csv")
    i_EM = 0
    i_SEM = 0


    for i in range(143):

        m = pd.read_csv("./data/"+ data + "/" + file(i) + ".csv",names=np.arange(76))
        row = []
        for y in range(76):
            row.extend(list(m[y]))
        if(res["mstype"][i] == 0):
            df_SEM.loc[str(i_SEM)] = row
            i_SEM+=1
        else:
            df_EM.loc[str(i_EM)] = row
            i_EM+=1

        df.loc[str(i)] = row
    
    df["mstype"] = list(res["mstype"])
    df.to_csv("./data/"+ data + "/" + data + ".csv")
    df_SEM.to_csv("./data/"+ data + "/" + data +'_SEM' + ".csv")
    df_EM.to_csv("./data/"+ data + "/" + data +'_EM' + ".csv")

    print(df)
    print("--")
    print(df_SEM)
    print("--")
    print(df_EM)


def configuration_graf(data,meth):

    rang = np.arange(76)
    df = pd.DataFrame(columns=rang)
    df_EM = pd.DataFrame(columns=rang)
    df_SEM = pd.DataFrame(columns=rang)

    res = pd.read_csv("./data/demographics.csv")
    i_EM = 0
    i_SEM = 0

    for i in range(143):
        m = pd.read_csv("./data/"+ data + "/" + file(i) + ".csv",names=np.arange(76))
        G = nx.from_numpy_matrix(m.to_numpy())

        row = []
        if meth == 1:
            r = G.degree()
        elif meth == 2:
            r = strength(m)
        elif meth == 3:
            r = nx.betweenness_centrality(G,weight='weight')
        elif meth == 4:
            r = nx.closeness_centrality(G,distance="weight")
        else:
            print('error')
            return

        for y in range(76):
            row.append(r[y])
        if(res["mstype"][i] == 0):
            df_SEM.loc[str(i_SEM)] = row
            i_SEM+=1
        else:
            df_EM.loc[str(i_EM)] = row
            i_EM+=1

        df.loc[str(i)] = row

    df["mstype"] = list(res["mstype"])
    print(df)
    # df.to_csv("./data/"+ data + "/" + data + '_' + str(meth) + "_graph.csv")


def file(number):
    if(number<10):
        return("000" + str(number))
    else:
        if(number<100):
            return("00"+ str(number))
        else:
            return("0"+ str(number))

def strength(matrix):
    l = []
    for i in range(76):
        sum = 0
        for y in range(76):
            if(i != y):
                sum += matrix[i][y]
        l.append(sum)
    return l
    