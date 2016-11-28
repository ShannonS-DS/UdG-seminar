import math
import numpy as np
from scipy.stats import wishart
from sklearn.cluster import KMeans
from scipy.special import psi, gammaln
from matplotlib.patches import Ellipse


def plot_cov_ellipse(cov, pos, nstd=2, ax=None, **kwargs):
    def eigsorted(cov):
        vals, vecs = np.linalg.eigh(cov)
        order = vals.argsort()[::-1]
        return vals[order], vecs[:,order]

    if ax is None:
        ax = plt.gca()

    vals, vecs = eigsorted(cov)
    theta = np.degrees(np.arctan2(*vecs[:,0][::-1]))

    # Width and height are "full" widths, not radius
    width, height = 2 * nstd * np.sqrt(vals)
    ellip = Ellipse(xy=pos, width=width, height=height, angle=theta, **kwargs)

    ax.add_artist(ellip)
    return ellip

def generate_data(N, K):
    a_pi = 10.        # Event proportions Dirichlet hyperparameter

    m_mu = np.zeros(2)      # Spatial component mean hyperparameter
    beta_mu = 0.001          # Spatial component mean hyperparameter
    W_Delta = 1.*np.eye(2)     # Spatial component precision hyperparameter
    nu_Delta = 10.          # Spatial component precision hyperparameter

    ln = np.zeros((N, 2))
    muk = np.zeros((K-1, 2))
    Deltak = np.zeros((K-1, 2, 2))

    en = np.zeros(N)

    for k in xrange(K-1):
        Deltak[k, :, :] = wishart.rvs(nu_Delta, W_Delta)
        muk[k, :] = np.random.multivariate_normal(m_mu, np.linalg.inv(beta_mu*Deltak[k, :, :]))

    alpha = [a_pi]*K
    alpha[K-1] *= 10
    pi = np.random.dirichlet(alpha)

    for n in xrange(N):
        ev = np.random.choice(K, 1, p=pi)[0]
        en[n] = ev
        if ev < (K-1):
            ln[n, :] = np.random.multivariate_normal(muk[ev, :],np.linalg.inv(Deltak[ev, :, :]))

    return ln, en

# Inference --------------------------------------------------------

def init_random_centers(K, N, xn):
    seed = np.random.RandomState()
    muk_ = np.array([seed.uniform(np.min(xn[:,0]),np.max(xn[:,1]),size=K),seed.uniform(np.min(xn[:,0]),np.max(xn[:,1]),size=K)]).T
    en_ = 0.1/(K-1)*np.ones((N, K))
    for n in xrange(N):
        en_[n,np.argmin([np.linalg.norm(xn[n,:]-muk_[k,:]) for k in xrange(K)])] = 0.9
    return en_

def init_kmeans(K, N,  xn):
    en_ = 0.1/(K-1)*np.ones((N, K))
    labels = KMeans(K).fit(xn).predict(xn)
    for i, lab in enumerate(labels):
        en_[i,lab] = 0.9
    return en_

def log_beta_function(x):
    return np.sum(gammaln(x + np.finfo(np.float32).eps))-gammaln(np.sum(x + np.finfo(np.float32).eps))

def lowerbound(ln, K, N, a_pi, pi_, en_,mk_, m_mu, betamuk_,beta_mu, nuk_, nu_Delta, Wk_, W_Delta):
    E3 = E2 = H2 =  0

    xk, yk = np.max(ln, axis=0)

    K = len(pi_)
    arr_a_pi = np.array([a_pi]*K)
    E1 = -log_beta_function(arr_a_pi) + np.dot((arr_a_pi-np.ones(K)), dirichlet_expectation(pi_))
    H1 = log_beta_function(pi_) - np.dot((pi_-np.ones(K)), dirichlet_expectation(pi_))

    logdet = np.log(np.array([np.linalg.det(Wk_[k,:,:]) for k in xrange(K)]))
    logDeltak = psi(nuk_/2.) + psi((nuk_-1.)/2.) + 2.*np.log(2.) + logdet
    for n in range(N):
        E2 += np.dot(en_[n,:], dirichlet_expectation(pi_))
        H2 += -np.dot(en_[n,:], np.log(en_[n,:]))
        product = np.array([np.dot(np.dot(ln[n,:]-mk_[k,:],Wk_[k,:,:]),(ln[n,:]-mk_[k,:]).T) for k in xrange(K)])
        E3 += 1./2*np.dot(en_[n,0:(K-1)],(logDeltak[0:(K-1)] -2.*np.log(2*math.pi) - nuk_[0:(K-1)]*product[0:(K-1)] - 2./betamuk_[0:(K-1)]).T)
        E3 += en_[n,(K-1)]*np.log(1./(xk*yk))

    product = np.array([np.dot(np.dot(mk_[k,:]-m_mu,Wk_[k,:,:]), (mk_[k,:]-m_mu).T) for k in xrange(K)])
    traces = np.array([np.matrix.trace(np.dot(np.linalg.inv(W_Delta), Wk_[k,:,:])) for k in xrange(K)])
    H4 = np.sum((1. + np.log(2.*math.pi) - 1./2*(np.log(betamuk_) + logdet))[0:(K-1)])
    logB = nuk_/2.*logdet + nuk_*np.log(2.) + 1./2*np.log(math.pi) + gammaln(nuk_/2.) + gammaln((nuk_-1)/2.)
    H5 = np.sum((logB - (nuk_-3.)/2.*logDeltak + nuk_)[0:(K-1)])
    E4 = np.sum((1./2*(np.log(beta_mu) + logDeltak - 2*np.log(2.*math.pi) - beta_mu*nuk_*product - 2.*beta_mu/betamuk_))[0:(K-1)])
    logB = nu_Delta/2.*np.log(np.linalg.det(W_Delta)) + nu_Delta*np.log(2.) + 1./2*np.log(math.pi) + gammaln(nu_Delta/2.) + gammaln((nu_Delta-1)/2.)
    E5 = np.sum((-logB + (nu_Delta-3.)/2.*logDeltak - nuk_/2.*traces)[0:(K-1)])

    return E1 + E2 + E3 + E4 + E4 + E5 + H1 + H2 + H4 + H5

def dirichlet_expectation(alpha):
    if len(alpha.shape) == 1:
        return psi(alpha + np.finfo(np.float32).eps) - psi(np.sum(alpha))
    return psi(alpha) - psi(np.sum(alpha, 1))[:, np.newaxis]

def initialization(ln, K, N, a_pi, m_mu, beta_mu, nu_Delta, W_Delta):
    bk_ = np.zeros(K)
    Wk_ = np.zeros((K, 2, 2))
    Selk = np.zeros((K, 2, 2))

    xk, yk = np.max(ln, axis=0)

    en_ = init_random_centers(K, N, ln)
    en_ = init_kmeans(K, N,ln)
    pi_ = a_pi + np.sum(en_, axis=0)
    Nek = np.sum(en_, axis=0)
    betamuk_ = beta_mu + Nek
    nuk_ = nu_Delta + Nek
    lk_ = np.tile(1./Nek,(2,1)).T * np.dot(en_.T,ln)
    mk_ = np.tile(1./betamuk_,(2,1)).T * (m_mu*beta_mu + np.dot(en_.T,ln))

    for k in xrange(K):
        Selk[k, :, :] = 1./Nek[k]*np.dot((ln-lk_[k,:]).T,np.dot(np.diag(en_[:,k]),(ln-lk_[k,:])))
        Wk_[k,:,:] = np.linalg.inv(np.linalg.inv(W_Delta)+Nek[k]*Selk[k,:,:] + beta_mu*Nek[k]/(beta_mu + Nek[k])*np.dot((lk_[k,:]-m_mu),(lk_[k,:]-m_mu).T))

    return pi_, en_, mk_, betamuk_, nuk_, Wk_

def variational_inference(It, ln, K, N, a_pi, m_mu, beta_mu, nu_Delta, W_Delta, pi_, en_, mk_, betamuk_, nuk_, Wk_):
    it = 0
    xk, yk = np.max(ln, axis=0)
    Selk = np.zeros((K, 2, 2))
    lwbound_old = -1000.
    lwbound = lowerbound(ln, K, N, a_pi, pi_,en_, mk_, m_mu, betamuk_, beta_mu, nuk_, nu_Delta, Wk_, W_Delta)
    print "Iteration: " + str(it) + " Lowerbound: " + str(lwbound) + " Diff: " + str(-(lwbound-lwbound_old)/lwbound)
    it +=1
    bounds = []
    bounds.append(lwbound)
    while it<It and abs((lwbound-lwbound_old)/lwbound)>1e-6 :
        pi_ = a_pi + np.sum(en_, axis=0)
        Elogpi_ = dirichlet_expectation(pi_)
        for n in xrange(N):
            aux = np.exp(Elogpi_)
            for k in xrange(K-1):
                aux[k] *= np.exp(-np.log(2*math.pi) + 1./2*(2*np.log(2) + psi(nuk_[k]/2.)+psi((nuk_[k]-1.)/2.)+np.log(np.linalg.det(Wk_[k,:,:]))-2./betamuk_[k]-nuk_[k]*(np.dot((ln[n,:]-mk_[k,:]).T,np.dot(Wk_[k,:,:],(ln[n,:]-mk_[k,:]))))))
            aux[K-1] *= 1./(xk*yk)
            en_[n, :] = aux/np.sum(aux) + np.finfo(np.float32).eps

        Nek = np.sum(en_, axis=0)
        lk_ = np.tile(1./Nek,(2, 1)).T * np.dot(en_.T, ln)
        betamuk_ = beta_mu + Nek
        nuk_ = nu_Delta + Nek
        mk_ = np.tile(1./betamuk_,(2,1)).T * (m_mu*beta_mu + np.dot(en_.T,ln))
        for k in xrange(K):
            Selk[k, :, :] = 1./Nek[k]*np.dot((ln-lk_[k,:]).T,np.dot(np.diag(en_[:,k]),(ln-lk_[k,:])))
            aux2 = np.linalg.inv(W_Delta)+Nek[k]*Selk[k,:,:] + beta_mu*Nek[k]/(beta_mu + Nek[k])*np.dot(np.tile((lk_[k,:]-m_mu),(1,1)).T,np.tile(lk_[k,:]-m_mu,(1,1)))
            Wk_[k, :, :] = np.linalg.inv(aux2)

        lwbound_old = np.copy(lwbound)
        lwbound = lowerbound(ln, K, N, a_pi, pi_,en_, mk_, m_mu, betamuk_, beta_mu, nuk_, nu_Delta, Wk_, W_Delta)
        print "Iteration: " + str(it) + " Lowerbound: " + str(lwbound) + " Diff: " + str(-(lwbound-lwbound_old)/lwbound)
        it +=1
        bounds.append(lwbound)

    return pi_, en_, mk_, betamuk_, nuk_, Wk_, bounds

