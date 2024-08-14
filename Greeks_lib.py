import numpy as np
from scipy.stats import norm

class Greeks_Stocks:
    def __init__(self,t):
        self.t = t
        
    def BSPrice(self,S,K,sig,r,q,T,w):
        tau = T - self.t
        d = (np.log(S/k) + r - q)/(sig*np.sqrt(tau))
        d1 = d + 0.5*sig*np.sqrt(tau)
        d2 = d - 0.5*sig*np.sqrt(tau)
        return w*S*np.exp(-q*tau)*norm.cdf(d1) - w*K*np.exp(-r*tau)*norm.cdf(d2)

    def Spot_delta(self,S,K,sig,r,q,T,w):
        tau = T - self.t
        d = (np.log(S/K) + r - q)/(sig*np.sqrt(tau))
        d1 = d + 0.5*sig*np.sqrt(tau)
        return np.exp(-q*tau)*w*norm.cdf(w*d1)

    def Strike_delta(self,S,K,sig,r,q,T,w):
        tau = T - self.t
        d = (np.log(S/K) + r - q)/(sig*np.sqrt(tau))
        d1 = d + 0.5*sig*np.sqrt(tau)
        return -np.exp(-r*tau)*w*norm.cdf(w*d1)

    def vega(self,S,K,sig,r,q,T):
        tau = T - self.t
        d = (np.log(S/K) + r - q)/(sig*np.sqrt(tau))
        d1 = d + 0.5*sig*np.sqrt(tau)
        return S*np.exp(-q*tau)*norm.pdf(d1)*np.sqrt(tau)

    def theta(self,S,K,sig,r,q,T,w):
        tau = T - self.t
        d = (np.log(S/K) + r - q)/(sig*np.sqrt(tau))
        d1 = d + 0.5*sig*np.sqrt(tau)
        d2 = d - 0.5*sig*np.sqrt(tau)
        return -0.5*np.exp(-q*tau)*(S*norm.pdf(d1)*sig)/np.sqrt(tau) -r*w*K*np.exp(-r*tau)*norm.cdf(w*d2) + q*w*S*np.exp(-q*tau)*norm.cdf(w*d1)

    def Gamma(self,S,K,sig,r,q,T):
        tau = T - self.t
        d = (np.log(S/K) + r - q)/(sig*np.sqrt(tau))
        d1 = d + 0.5*sig*np.sqrt(tau)
        return np.exp(-q*tau)*norm.pdf(d1)/(S*sig*np.sqrt(tau))