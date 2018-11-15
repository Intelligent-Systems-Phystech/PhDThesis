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

plt.plot(x,y_test,'b-',label=u'Ошибка на контрольной выборке, Theano')#,yerr=ytest_err,label)
plt.fill_between(x, y_test_all1, y_test_all2,color='b', alpha=0.2)
plt.plot(x,y_train,'g-',label=u'Ошибка на  обучающей выборке, Theano')#,yerr=ytest_err,label)
plt.fill_between(x, y_train_all1, y_train_all2,color='g', alpha=0.2)

y_trainedge1 = [0.1986 ,  0.1733 ,  0.1533 ,  0.1347 ,  0.1294 ,  0.1233 ,
        0.10286,  0.096  ,  0.09571,  0.0886 ,  0.0735 ,  0.0725 ,  0.071,0.071]
y_trainedge2 = [ 0.16429,  0.14857,  0.11143,  0.0945 ,  0.07571,  0.07   ,
        0.07143,  0.07   ,  0.05286,  0.05   ,  0.0439 ,  0.0457 ,  0.0437,  0.0437 ]
y_testedge1 = [0.235  ,  0.2244 ,  0.2117 ,  0.1919 ,  0.1749 ,  0.15554,
        0.16418,  0.1506 ,  0.15418,  0.15771,  0.15277,  0.1506 ,  0.14458,0.14458]
y_testedge2 = [ 0.1015,0.1015 ,  0.10606,  0.1115 ,  0.11212,  0.12424,  0.12651,
        0.1303 ,  0.1473 ,  0.1572 ,  0.1621 ,  0.1735 ,  0.1923 ,  0.21515][::-1]
#y_ml_train = [0.1986 ,  0.1733 ,  0.1533 ,  0.1347 ,  0.1294 ,  0.1233 ,
        #0.10286,  0.096  ,  0.09571,  0.0886 ,  0.0735 ,  0.0725 ,  0.071,  0.071]
y_ml_train =[ 0.18132 ,  0.157226,  0.136226,  0.116091,  0.105397,  0.098511,
        0.083434,  0.081769,  0.079026,  0.070006,  0.061674,  0.064771,
        0.059429,0.059429 ]
y_ml_test = [0.2237 ,  0.21004,  0.18862,  0.1766 ,  0.16163,  0.15034,
        0.14914,  0.14145,  0.14235,  0.13964,  0.13423,  0.13313,  0.13126,0.13126]
x_ml = []
for i in range(100,750,50):
    rbm = i*10/int(19)
    ae = i*6/int(19)
    sm = i*3/int(19)
    x_ml.append(600*rbm+rbm + rbm*ae+ae + ae*sm + sm + sm*6 + 6) 
x_ml.append(x[-1])
plt.plot(x_ml,y_ml_train,'r-',label=u'Ошибка на обучающей выборке, Matlab')
plt.fill_between(x_ml, y_trainedge1, y_trainedge2, color='r', alpha=0.2)
plt.plot(x_ml,y_ml_test,'k-',label=u'Ошибка на контрольной выборке, Matlab')
plt.fill_between(x_ml, y_testedge1, y_testedge2, color='k', alpha=0.2)

#plt.plot(x_ml,y_edge,'k--',label=u'test')
#plt.plot(x_ml,y_edge2,'k--',label=u'test')
plt.legend(loc='upper right')
plt.xlabel(u'Количество параметров')
plt.ylabel(u'Средняя ошибка')
"""

#plt.errorbar(x,y_train,yerr=ytrain_err)
x=[10,20,30,40,50,60,70,75]

y_test = [ 0.36367628607277291,0.27879548306148055 ,0.2259723964868256,0.19824341279799249,0.16173149309912171,0.15257214554579673, 0.14385194479297364, 0.12961104140526974]
ytest_err = [0.016403320414567782,0.011739039032098844,0.02269820213659899,0.013804607895276476,0.0095609025972108342,0.00853199498117942,0.011940482441863274,0.0087095879743039964]

y_train = [0.11215686274509803,0.094274509803921575, 0.10117647058823528,0.096470588235294114,0.092517647058823524,0.088705882352941162,0.091831932773109234,0.085672453461618919]
ytrain_err = [0.011322404663603824,0.0039419080386287218,0.017708438401961936,0.012243838143005899,0.0086467662571109993,0.0053646396306220095,0.0070027921568570064,0.0085606513691373395]

y_test_all1 = np.array(y_test) -  np.array(ytest_err)
y_test_all2 = np.array(y_test) +  np.array(ytest_err)
y_train_all1 = np.array(y_train) -  np.array(ytrain_err)
y_train_all2 = np.array(y_train) +  np.array(ytrain_err)


y_test_upper = [0.0689169, 0.0859792, 0.101558, 0.0859792, 0.100074, 0.104154, 0.0959941, 0.0937685]
y_test_lower = [0.0626113, 0.0711424, 0.0941395, 0.0737389, 0.0908012, 0.0848665, 0.087092, 0.0848665]
y_train_lower = [0.242136, 0.224703, 0.196884, 0.180193, 0.177596, 0.163131, 0.141988, 0.130119]
y_train_upper = [0.26773, 0.241766, 0.210608, 0.199481, 0.192062, 0.179451, 0.150519, 0.13865]


y_train_ml = [  0.253567, 0.235787, 0.206125, 0.189833, 0.18691, 0.172108, 0.149872, 0.13727  ]
y_test_ml = [0.0659496, 0.0815282, 0.0971068, 0.08227, 0.0952522, 0.091543, 0.0911721, 0.0896884 ]
plt.plot(x,y_test,'b-',label=u'Ошибка на контрольной выборке, Theano')#,yerr=ytest_err,label)
plt.fill_between(x, y_test_all1, y_test_all2, alpha=0.1)
plt.plot(x,y_train,'g-',label=u'Ошибка на обучающей выборке, Theano')#,yerr=ytest_err,label)
plt.fill_between(x, y_train_all1, y_train_all2, alpha=0.1)
plt.plot(x,y_train_ml,'r-',label=u'Ошибка на обучающей выборке, Matlab')#,yerr=ytest_err,label)
plt.plot(x,y_test_ml,'k-',label=u'Ошибка на контрольной выборке, Matlab')#,yerr=ytest_err,label)
plt.fill_between(x, y_test_lower, y_test_upper, color='k', alpha=0.2)
plt.fill_between(x, y_train_lower, y_train_upper, color='r', alpha=0.2)

plt.xlim((10,75))
plt.legend(loc='upper right')
plt.xlabel(u'Процент выборки')
plt.ylabel(u'Средняя ошибка')
"""
plt.xlim((50000, x[-1]))
plt.show()
