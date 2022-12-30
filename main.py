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
    t_inicio  = datetime.now()

    # Crate the max matrix
    configuration_matrix(data)
    for i in range(1,6):
        configuration_graf(data,i)

    t_final = datetime.now()
    time = t_final - t_inicio
    seconds = time.seconds
    print(str(seconds) + "s")
    
    #Descrip the data base and crate the t_studen & PCA matrics
    datapreparation_matrix(data,False,0)
    for i in range(1,6):
        datapreparation_matrix(data,True,i)

    #Entrenar modelos
    K_NN("t_student","PCA_funtion",data,True, 1)
    decisionTree("t_student","PCA_funtion",data, True, 1)
    svm("t_student","PCA_funtion",data,True, 1)    
    randomForest("t_student","PCA_funtion",data,True, 1)
    RN("t_student","PCA_funtion",data,True, 1)

    for i in range(1,5):
        K_NN("t_student","PCA_funtion",data,True, i)
        decisionTree("t_student","PCA_funtion",data, True, i)
        svm("t_student","PCA_funtion",data,True, i)    
        randomForest("t_student","PCA_funtion",data,True, i)
        RN("t_student","PCA_funtion",data,True, i)


if __name__ == '__main__':
    sys.exit(main())



#{
#  n:Nodo - 1
#  Inice : n*76 --> n*76+n
# }

