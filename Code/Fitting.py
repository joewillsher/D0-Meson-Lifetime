from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, hbar, physical_constants
import pylab as pl
import scipy.optimize as spo
import lazy_property
from Lifetime import *
from Cuts import filtered

times = [d.decayTime*1e12 for d in filtered if 10e-12 >= d.decayTime] #decay times considered


##############################################
#estimations of fitting parameters:
hist, bin_edges = np.histogram(times, bins=100, range=(0.1, 10))
time = bin_edges[1:]

ML = np.mean(times) #analytic mean lifetime by likelihood maximisation

first_lifetime_t = np.array([t for t in time if t < ML]) #selects the area of data to look at for fitting
first_lifetime_hist = np.array(hist[:len(first_lifetime_t)]) #corresponding histogram values

Amp_int = first_lifetime_hist * np.exp(first_lifetime_t/ML)
Amp_avg = np.mean(Amp_int) #average amplitude

print('Amplitude', Amp_avg, 'Mean lifetime', ML)

#plots fit + data on the same graph
newfig()
pl.plot(time, hist, '-b')
pl.plot(time, Amp_avg * np.exp(-time/ML), '-r')
pl.xlabel(r'Decay time / ps')
pl.savefig('decay-fitted-ML')
pl.close()

#chi squared test:
expected_value = Amp_avg * np.exp(-time/ML)
observed_value = hist
diff_sqrd = (observed_value - expected_value)**2
chi_int = diff_sqrd/expected_value

pl.plot(time, chi_int)
pl.savefig('chi_squared vs time')
pl.close()

chi_squared = sum(chi_int)
mean_chi_squared = np.mean(chi_int)

#Residual sum of squares:
RSS_int =diff_sqrd
print("max RSS", max(RSS_int))
RSS = sum(diff_sqrd)
mean_RSS= np.mean(RSS_int)

pl.plot(time, RSS_int)
pl.savefig('RSS vs time')
pl.close()

print('chi_squared', chi_squared, 'mean_chi_squared', mean_chi_squared, 'RSS', RSS, 'mean_RSS', mean_RSS)
#########################################

N=len(times)
T=sum(times)
range_tau = np.linspace(0.3, 0.7, 1000)

def negative_log_likelihood(tau):
    return N*np.log(tau) + T/tau

def DL_Dtau(x):
    return (N/x) - (T/(x**2))

def DL_2_Dtau(x):
    return -N/(x**2) + (2*T)/(x**3)

L=negative_log_likelihood(range_tau)
DL=DL_Dtau(range_tau)

pl.plot(range_tau, DL)
pl.savefig('DL vs tau')
pl.close()

#pl.plot(range_tau, L)
#pl.savefig('L vs tau')
tau_0 = 0.3
tau = tau_0

#def convoluted_exponential(t, A, l, s, m):
	#if s < 0 and m == 0:
		#return 0
	#return A* l/2 * np.exp(2*m + l * s**2 - 2*t) * sse.erfc((m + l * s**2 - t)/(2**0.5 * s))
def Newton_Raphson_tau(tau):
    while np.abs(DL_Dtau(tau)) >= 0.005:
        tau_n = tau - DL_Dtau(tau)/DL_2_Dtau(tau)
        tau = tau_n
    return tau

tau_f = Newton_Raphson_tau(tau)
print ('tau_f', tau_f)

thresh = negative_log_likelihood(tau_f) +1

def Neg_Log_1(tau):
    return N*np.log(tau) + T/tau - thresh

NL_1 = Neg_Log_1(range_tau)

def Newton_Raphson_uncertainty(x):
    while np.abs(Neg_Log_1(x)) >= 0.005:
        x_n = x - Neg_Log_1(x)/DL_Dtau(x)
        x=x_n
    return x

x_1= Newton_Raphson_uncertainty(tau_f - 0.02)
x_2 = Newton_Raphson_uncertainty(tau_f + 0.02)

print ('X_1, X_2', x_1, x_2)

def error(x_1, x_2, x):
    sigma = np.abs(x-x_1) + np.abs(x-x_2)
    return sigma

S = 0.5 * error(x_1, x_2, tau_f)

print('lifetime ', tau_f, '+- ', S, ' ps')
