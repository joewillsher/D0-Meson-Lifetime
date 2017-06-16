from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, hbar, physical_constants
from scipy.misc import derivative
import pylab as pl
import scipy.optimize as spo
import lazy_property
from Background import *


def maximum_likelyhood_exp_fit(full_set, after_po, deltamass_peak_width):
	range_low, range_up = get_sig_range(after_po, deltamass_peak_width)
	print(range_low, range_up)
	wb = calculate_weight(after_po, full_set, range_low, range_up)
	
	times = [d.decayTime*1e12 for d in full_set if 10e-12 >= d.decayTime] #decay times considered from full_set
	signal_region = [event for event in full_set if range_low <= event.massDiff_d0dstar <= range_up] #events in the signal region
	times_sig = [d.decayTime*1e12 for d in signal_region] #decay times from events in the signal region
	background_region = [event for event in full_set if event not in signal_region] #events in the background sideband
	times_background = [d.decayTime*1e12 for d in background_region] #decay times from events in the sideband region

	weighted_decays = [i * wb for i in times_background] + times_sig #decay times with weight applied to events in the sideband region. Not used. Is it useful?

	#Negative log likelihood minimisation:
	range_tau = np.linspace(0.3, 0.7, 1000) #range of mean lifetime considered for minimisation

	def negative_log_likelihood(ts, l): #t is decay times, l is lifetime
		# the individual log likelyhoods		
		L_i = [(convoluted_exponential_1(x, 1, l, 0.5, 0)) ** (1 if range_low <= x.massDiff_d0dstar <= range_up else wb) for x in ts]
		L = L_i.prod() #multiplies all the elements together WARNING: at this point, it still should be a function of tau
		neg_log = - np.ln(L) #takes the negative of the log. NOW it should substitute tau
		return neg_log

	def D_Dtau(x_0): #first derivative wrt tau
		DL_1 = []
		for t in x_0: #x_0 is range of tau
			DL = derivative(neg_log, t) #calculates the function at each point in range_tau
			DL_1 = DL_1.append(DL)
		return DL_1

	def D_2_Dtau(x_0):
		DL_2=[]
		for t in x_0:
			DL = derivative(neg_log, t, order=2)
			DL_2 = DL_2.append(DL)
		return DL_2

	NL = negative_log_likelihood(times, range_tau)
	D_1 = D_Dtau(range_tau)
	D_2 = D_2_Dtau(range_tau)

	###plots to check the function above are working properly
	pl.plot(range_tau, NL)
	pl.savefig('L vs tau')
	pl.close()
	pl.plot(range_tau, D_1)
	pl.savefig('DL vs tau')
	pl.close()
	pl.plot(range_tau, D_2)
	pl.savefig('D2L vs tau')
	pl.close()

	def Newton_Raphson_tau(tau): #iterates and finds the root of the derivative (minimum of log likelihood)
		while np.abs(D_Dtau(tau)) >= 0.005: #precision to reach
			tau_n = tau - D_Dtau(tau)/D_2_Dtau(tau)
			tau = tau_n
		return tau #returns the root

	#statistical uncertainty calculations

	def Newton_Raphson_uncertainty(x): #finds the root of Neg_Log_1
		while np.abs(Neg_Log_1(x)) >= 0.005:
			x_n = x - Neg_Log_1(x)/DL_Dtau(x)
			x=x_n
		return x

	#statistical uncertainty calculations
	def Neg_Log_1(tau): #Neg Log likelihood shifted by a threshold
		return negative_log_likelihood(times, tau) - negative_log_likelihood(times, tau_f) - 1 #WARNING: tau_f is a number, not an array, negative_log_likelihood might crash


	def Newton_Raphson_uncertainty(x): #finds the root of Neg_Log_1
		while np.abs(Neg_Log_1(x)) >= 0.005:
			x_n = x - Neg_Log_1(x)/D_Dtau(x)
			x=x_n
		return x

	tau_f = Newton_Raphson_tau(0.3)
	print ('tau_f', tau_f)
	NL_1 = Neg_Log_1(range_tau)
	
	x_1= Newton_Raphson_uncertainty(tau_f - 0.02) #finds the two roots of Neg_Log_1
	x_2 = Newton_Raphson_uncertainty(tau_f + 0.02)

	S = error(x_1, x_2, tau_f)

	S = error(x_1, x_2, tau_f)

	print('lifetime ', tau_f, '+- ', S, ' ps')
	print('x1, x2', x_1, x_2)
