from re import I
import pandas as pd
import numpy as np


def configuration(data):
    
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



def file(number):
    if(number<10):
        return("000" + str(number))
    else:
        if(number<100):
            return("00"+ str(number))
        else:
            return("0"+ str(number))