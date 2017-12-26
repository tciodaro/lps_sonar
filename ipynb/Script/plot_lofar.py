
import SonarData as sd
import numpy as np
from scipy import linalg
from scipy import signal
import matplotlib.pyplot as plt



ship = 'ClasseA'
run = 0
lofarParam = {
                'Overlap' : 0,
                'NFFT' : 2048
             }
##########################################################################################
## DATA OBJECT
data = sd.SonarData(ship, run)
data.LOFAR(param=lofarParam)
plt.figure(figsize=(8,8))
data.PlotLOFAR(ship, run)
plt.show()
