# -*- coding: utf-8 -*-
__author__ = 'bahteev'
from matplotlib import pyplot as plt
import numpy as np
from matplotlib import rc
font = {'family':  "DejaVu Sans",
        'weight': 'normal',
        'size': 20}

rc('font', **font)

x_rbm = [50,75,112,168,252,378]
x_ae = [30,45,67,100,150,225]
x_sm = [15,23,35,52,78,117]
x = []
for i in range(0,len(x_sm)):
    x.append(600*x_rbm[i]+x_rbm[i] + x_rbm[i]*x_ae[i]+x_ae[i] + x_ae[i]*x_sm[i] + x_sm[i] + x_sm[i]*6 + 6) 
#x = [95, 74+45+23,112+67+35,168+100+52,252+150+78,378+225+117]
y_test = [0.29667503136762863,  0.26016311166875783, 0.18889585947302384, 0.16900878293601004, 0.1468005018820577, 0.12961104140526974]
ytest_err = [0.012147658391296055,0.02998824041006572,0.010507734218784604,0.020562814410632152,0.007459958090938394,0.0087095879743039964]
y_train = [0.25015687094750055, 0.21899184271072997, 0.13969880778079899, 0.12390713239907969, 0.09627692951265425, 0.085672453461618919]
ytrain_err = [0.011315798323903292,0.032224888754107048,0.0073731644579878011,0.021293027529300325,0.0069457917594393671,0.0085606513691373395]
y_test_all1 = np.array(y_test) -  np.array(ytest_err)
y_test_all2 = np.array(y_test) +  np.array(ytest_err)
y_train_all1 = np.array(y_train) -  np.array(ytrain_err)
y_train_all2 = np.array(y_train) +  np.array(ytrain_err)

plt.plot(x,y_test,'b-',label=u'Ошибка на контрольной выборке')#,yerr=ytest_err,label)
plt.fill_between(x, y_test_all1, y_test_all2, alpha=0.1)
plt.plot(x,y_train,'g-',label=u'Ошибка на  обучающей выборке')#,yerr=ytest_err,label)
plt.fill_between(x, y_train_all1, y_train_all2, alpha=0.1)
plt.legend(loc='lower left')
plt.xlabel(u'Количество параметров')
plt.ylabel(u'Средняя ошибка')

plt.show()
