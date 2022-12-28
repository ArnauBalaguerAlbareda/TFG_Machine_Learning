# import pandas as pd
# import numpy as np
from confidata.DataSetConfi import *
from datetime import datetime
import sys
from datapreparation.datapreparation import *
from classificationmodel.K_nn import *
from classificationmodel.decisionTree import *
from classificationmodel.randmoForest import *
from classificationmodel.svm import *
from classificationmodel.RN import *


def main():

    data = 'RS'
    # t_inicio  = datetime.now()

    # configuration_matrix(data)
    configuration_graf(data,1)

    # t_final = datetime.now()
    # time = t_final - t_inicio
    # seconds = time.seconds
    # print(str(seconds) + "s")
    
    # datapreparation_matrix(data,True,1)

    # K_NN("t_student","PCA_funtion",data,True, 1)
    # decisionTree("t_student","PCA_funtion",data, True, 1)
    # svm("t_student","PCA_funtion",data,True, 1)    
    # randomForest("t_student","PCA_funtion",data,True, 1)
    # RN("t_student","PCA_funtion",data,True, 1)



if __name__ == '__main__':
    sys.exit(main())



#{
#  n:Nodo - 1
#  Inice : n*76 --> n*76+n
# }

