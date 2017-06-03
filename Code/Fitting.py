from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, hbar, physical_constants
import pylab as pl
import scipy.optimize as spo
import lazy_property
from Lifetime import *
from Cuts import *

times = [d.decayTime*1e12 for d in filtered if 10e-12 >= d.decayTime] #decay times considered

ML = np.mean(times) #analytic mean lifetime
print('Mean lifetime', ML)

#Negative log likelihood minimisation:
N=len(times) #number of decays
T=sum(times) #sum of all decay times
range_tau = np.linspace(0.3, 0.7, 1000) #range of mean lifetime considered for minimisation

def negative_log_likelihood(tau): #negative log likelihood as a function of tau
    return N*np.log(tau) + T/tau

def DL_Dtau(x): #first derivative wrt tau
    return (N/x) - (T/(x**2))

def DL_2_Dtau(x): #second derivative wrt tau
    return -N/(x**2) + (2*T)/(x**3)

#def convoluted_exponential(t, A, l, s, m):
	#if s < 0 and m == 0:
		#return 0
	#return A* l/2 * np.exp(2*m + l * s**2 - 2*t) * sse.erfc((m + l * s**2 - t)/(2**0.5 * s))

def Newton_Raphson_tau(tau): #iterates and finds the root of the derivative (minimum of log likelihood)
    while np.abs(DL_Dtau(tau)) >= 0.005: #precision to reach
        tau_n = tau - DL_Dtau(tau)/DL_2_Dtau(tau)
        tau = tau_n
    return tau #returns the root

tau_f = Newton_Raphson_tau(0.3)
print ('tau_f', tau_f)

#statistical uncertainty calculations

def Neg_Log_1(tau): #Neg Log likelihood shifted by a threshold
    return N*np.log(tau) + T/tau - negative_log_likelihood(tau_f) - 1

NL_1 = Neg_Log_1(range_tau)

def Newton_Raphson_uncertainty(x): #finds the root of Neg_Log_1
    while np.abs(Neg_Log_1(x)) >= 0.005:
        x_n = x - Neg_Log_1(x)/DL_Dtau(x)
        x=x_n
    return x

x_1= Newton_Raphson_uncertainty(tau_f - 0.02) #finds the two roots of Neg_Log_1
x_2 = Newton_Raphson_uncertainty(tau_f + 0.02)

def error(x_1, x_2, x): #returns the uncertainty on tau
    sigma = 0.5*(np.abs(x-x_1) + np.abs(x-x_2))
    return sigma

S = error(x_1, x_2, tau_f)

print('lifetime ', tau_f, '+- ', S, ' ps')
print('x1, x2', x_1, x_2)

#wb = calculate_weight(after_po, filtered, after_bin_width)
#print (wb)
