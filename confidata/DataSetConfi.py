from re import I
import pandas as pd
import numpy as np


def configuration(data):
    
    rang = np.arange(76*76)
    df = pd.DataFrame(columns=rang)
    res = pd.read_csv("./data/demographics.csv")

    for i in range(143):

        m = pd.read_csv("./data/"+ data + "/" + file(i) + ".csv",names=rang)
        row = []
        for y in range(76):
            row.extend(list(m[y]))

        df.loc[str(i)] = row
    
    df["mstype"] = list(res["mstype"])
    df.to_csv(data + '.csv')

    print(df)


def file(number):
    if(number<10):
        return("000" + str(number))
    else:
        if(number<100):
            return("00"+ str(number))
        else:
            return("0"+ str(number))