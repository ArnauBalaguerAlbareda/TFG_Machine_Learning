# import pandas as pd
# import numpy as np
from confidata.DataSetConfi import *
from datetime import datetime
import sys
from datapreparation.datapreparation import *
from classificationmodel.K_nn import *


def main():

    data = 'FA'
    # t_inicio  = datetime.now()

    # configuration(data)

    # t_final = datetime.now()
    # time = t_final - t_inicio
    # seconds = time.seconds
    # print(str(seconds) + "s")
    
    # datapreparation(data)

    K_NN("hola","PCA_funtion",data)






if __name__ == '__main__':
    sys.exit(main())



#{
#  n:Nodo - 1
#  Inice : n*76 --> n*76+n
# }

