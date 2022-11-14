# import pandas as pd
# import numpy as np
from confidata.DataSetConfi import *
from datetime import datetime
import sys
from datapreparation.datapreparation import *


def main():
    t_inicio  = datetime.now()

    configuration("FA")

    t_final = datetime.now()
    time = t_final - t_inicio
    seconds = time.seconds
    print(str(seconds) + "s")
    datapreparation('FA')




if __name__ == '__main__':
    sys.exit(main())



#{
#  n:Nodo - 1
#  Inice : n*76 --> n*76+n
# }

