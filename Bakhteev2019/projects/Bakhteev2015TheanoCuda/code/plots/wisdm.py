# -*- coding: utf-8 -*-
import numpy as np
data = np.load('../../data/WISDM.npy')
row_walk, row_run, row_sit = None,None,None
for row in data:
	if row[0]==0:
		row_walk = row[1:]
	if row[0]==1:
		row_run = row[1:]
	if row[0]==2:
		row_sit = row[1:]
	
	if row_walk is not None	 and row_run is not None and row_sit is not None:
		print 'found all'
		break

from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rc
font = {'family':  "DejaVu Sans",
        'weight': 'normal',
        'size': 20}
rc('font', **font)
i_y = [i for i in range(0, len(row_walk)) if i%3==1]
plt.plot([i for i in range(0, len(i_y))], row_walk[i_y],label=u'Ходьба')
plt.plot([i for i in range(0, len(i_y))], row_run[i_y]-20,label=u'Бег')
plt.plot([i for i in range(0, len(i_y))], row_sit[i_y]-40,label=u'Сидение')
plt.xlabel(u'Время')
plt.ylabel(u'Данные с акселерометра')
plt.legend(loc='lower right')
plt.show()
