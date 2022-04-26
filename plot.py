import matplotlib.pyplot as plt
import numpy as np

x = [5,15,25]
OMP_cropped = [60,78.6, 92]
FISTA_cropped = [59,81,90.33]
OMP_original = [39.4,43.2,50.4]
FISTA_original = [20.6,30,35]
plt.plot(x, OMP_cropped, '-',label='OMP_cropped')
plt.plot(x, FISTA_cropped,'--',label='FISTA_cropped')
plt.plot(x, OMP_original,'-.',label='OMP_original')
plt.plot(x, FISTA_original,':',label='FISTA_original')
plt.legend(loc = 'upper left',fontsize='x-small')
plt.xlabel('# of measurements')
plt.ylabel('Accuracy')
plt.show()
