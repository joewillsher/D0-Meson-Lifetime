from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, hbar, physical_constants
import pylab as pl
import scipy.optimize as spo
import lazy_property


def maximum_likelyhood_exp_fit(times, wb):
	N=len(times) #number of decays
	T=sum(times) #sum of all decay times
	#Negative log likelihood minimisation:
	range_tau = np.linspace(0.3, 0.7, 1000) #range of mean lifetime considered for minimisation
	
	def negative_log_likelihood(tau): #negative log likelihood as a function of tau
		return N*np.log(tau) + T/tau

	def DL_Dtau(x): #first derivative wrt tau
		return (N/x) - (T/(x**2))

	def DL_2_Dtau(x): #second derivative wrt tau
		return -N/(x**2) + (2*T)/(x**3)

	def Newton_Raphson_tau(tau): #iterates and finds the root of the derivative (minimum of log likelihood)
		while np.abs(DL_Dtau(tau)) >= 0.005: #precision to reach
			tau_n = tau - DL_Dtau(tau)/DL_2_Dtau(tau)
			tau = tau_n
		return tau #returns the root

	#statistical uncertainty calculations

	def Neg_Log_1(tau): #Neg Log likelihood shifted by a threshold
		return N*np.log(tau) + T/tau - negative_log_likelihood(tau_f) - 1


	def Newton_Raphson_uncertainty(x): #finds the root of Neg_Log_1
		while np.abs(Neg_Log_1(x)) >= 0.005:
			x_n = x - Neg_Log_1(x)/DL_Dtau(x)
			x=x_n
		return x


	def error(x_1, x_2, x): #returns the uncertainty on tau
		sigma = 0.5*(np.abs(x-x_1) + np.abs(x-x_2))
		return sigma



	tau_f = Newton_Raphson_tau(0.3)
	print ('tau_f', tau_f)
	NL_1 = Neg_Log_1(range_tau)
	
	x_1= Newton_Raphson_uncertainty(tau_f - 0.02) #finds the two roots of Neg_Log_1
	x_2 = Newton_Raphson_uncertainty(tau_f + 0.02)

	S = error(x_1, x_2, tau_f)

	print('lifetime ', tau_f, '+-', S, ' ps')
	print('x1, x2', x_1, x_2)
	print(wb)
