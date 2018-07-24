import theano
import theano.tensor as T
import numpy as np
import scipy.stats
def gaus_prior(params, log_alphas = theano.shared(np.array([0.0])), gaus_type='scalar'):
    
    if gaus_type not in ['scalar', 'diagonal']:
        raise ValueError('Bad gaus type vaule')
    if gaus_type=='scalar':          
           return ((np.log(2*np.pi) + (log_alphas[0])*2)*(-0.5 * params.shape[0]) - 0.5*T.dot(params/T.exp(log_alphas[0]), params.T/T.exp(log_alphas[0])))
    elif gaus_type=='diagonal':
        return ((np.log(2*np.pi) )*(-0.5 * params.shape[0]) + T.sum((log_alphas)*2)*(-0.5)   - 0.5*T.dot(params/T.exp(log_alphas), params.T/T.exp(log_alphas)))
    else:
        raise ValueError('Bad gaus prior type') 


def KLD(mu0, mu1, log_sigma_0, log_sigma_1, gaus_type='diagonal'):
    #Actually, not simga, but sigma^2. TODO: rename. The code is used properly. Check Var_feedforward.
    if gaus_type!='diagonal':
        raise ValueError('Bad gaus type')
    """
    see Russian wiki :)
    """
    part1 = T.sum(T.exp(log_sigma_0 - log_sigma_1))
    dif = (mu1-mu0)
    part2 = T.dot(dif.T/ T.exp(log_sigma_1), dif)
    part3 = -mu0.shape[0]
    #log(det(exp(A))) = tr(A)
    part4 = T.sum(log_sigma_1) - T.sum(log_sigma_0)
    return 0.5*(part1+part2+part3+part4)


#FOR TEST ONLY, http://www.cs.cmu.edu/~chanwook/MySoftware/rm1_Spk-by-Spk_MLLR/rm1_PNCC_MLLR_1/rm1/python/sphinx/divergence.py
def gau_kl(pm, pv, qm, qv):
    """
    Kullback-Liebler divergence from Gaussian pm,pv to Gaussian qm,qv.
    Also computes KL divergence from a single Gaussian pm,pv to a set
    of Gaussians qm,qv.
    Diagonal covariances are assumed.  Divergence is expressed in nats.
    """
    if (len(qm.shape) == 2):
        axis = 1
    else:
        axis = 0
    
    # Determinants of diagonal covariances pv, qv
    dpv = pv.prod()

    dqv = qv.prod(axis)

    # Inverse of diagonal covariance qv
    iqv = 1./qv
    # Difference between means pm, qm
    diff = qm - pm
    return (0.5 *
            (np.log(dqv / dpv)          # log |\Sigma_q| / |\Sigma_p|
             + (iqv * pv).sum(axis)         # + tr(\Sigma_q^{-1} * \Sigma_p)
             + (diff * iqv * diff).sum(axis) # + (\mu_q-\mu_p)^T\Sigma_q^{-1}(\mu_q-\mu_p)
                  - len(pm)))                     # - N
                

if __name__=='__main__':
      
       assert np.linalg.norm(gaus_prior(np.array([1,1]), log_alphas = theano.shared(np.array([np.log(2.0)])), gaus_type='scalar').eval() - np.log(np.prod(scipy.stats.norm.pdf([1,1], scale=[2.0, 2.0]))))<0.01
       
       assert np.linalg.norm(gaus_prior(np.array([1,1]), log_alphas  = theano.shared(np.log([2.0, 2.0])) , gaus_type='diagonal').eval() - ( np.log(np.prod(scipy.stats.norm.pdf([1,1], scale=[2.0, 2.0])))))<0.01
       assert np.linalg.norm(gaus_prior(np.array([1,1]), log_alphas  = theano.shared(np.log([2.0, 1.0])) , gaus_type='diagonal').eval() - ( np.log(np.prod(scipy.stats.norm.pdf([1,1], scale=[2.0, 1.0])))))<0.01
       



       
       assert KLD(np.array([1.0]), np.array([2.0]), np.array([-1]), np.array([-2])).eval() ==  gau_kl(np.array([1.0]), np.exp(np.array([-1])), np.array([2]),  np.exp(np.array([-2])))
      

       assert KLD(np.zeros(4),np.zeros(4),np.zeros(4),np.zeros(4)).eval()==0 
