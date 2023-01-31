import pandas as pd
from matplotlib import pyplot as plt
import numpy as np

from sklearn.model_selection import cross_val_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.utils import resample
from sklearn.metrics import precision_score,accuracy_score, recall_score ,f1_score
from sklearn import preprocessing


def seve_model( x, y,model, name_model, nameMatrix, graf, meth):

    lb = preprocessing.LabelBinarizer()
    y = lb.fit_transform(y).ravel()
    accuracy = (cross_val_score(model, x, y, cv = 7, scoring='accuracy' ).mean())
    precision =(cross_val_score(model, x, y, cv = 7, scoring="precision").mean())
    recall = (cross_val_score(model, x, y,scoring="recall", cv = 7).mean())
    f1 = (cross_val_score(model, x, y,scoring="f1", cv = 7).mean())

    # if(graf == False):joblib.dump(model,"./data/"+ nameMatrix + "/" + nameMatrix + 'model_' + name_model + '.pkl')
    # else: joblib.dump(model,"./data/"+ nameMatrix + "/" + nameMatrix + 'model_' + name_model + "_" + str(meth) + '.pkl')