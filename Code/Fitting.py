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
tau = np.linspace(0.3, 0.6, 1000)
negative_log_likelihood = N*np.log(tau) + T/tau
def DL_Dtau(x):
    return (N/x) - (T/(x**2))

def DL_2_Dtau(x):
    return -N/(x**2) + (2*T)/(x**3)

DL=DL_Dtau(tau)

pl.plot(tau, DL)
pl.savefig('DL vs tau')

tau_0 = 0.4
tau = tau_0

def mod(x) :
    if x < 0:
        return - x
    else:
        return x

while mod(DL_Dtau(tau)) >= 0.005:
    tau_n = tau - DL_Dtau(tau)/DL_2_Dtau(tau)
    tau = tau_n

print ('tau', tau )
