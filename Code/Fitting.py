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
from style import *


def maximum_likelyhood_exp_fit(full_set, after_po, deltamass_peak_width):

# 	np.save('fitting_FULLSET.npy', full_set)
# 	np.save('fitting_AFTERPO.npy', after_po)
# 	np.save('fitting_WIDTH.npy', [deltamass_peak_width])
	
	range_low, range_up = get_sig_range(after_po, deltamass_peak_width)
	print(range_low, range_up)
	
	data = [event for event in full_set if 0 <= event.decayTime <= 10e-12]

	wb = calculate_weight(after_po, data, range_low, range_up)
	
	times = [d.decayTime*1e12 for d in data] #decay times considered from data
	mass_diffs = [event.massDiff_d0dstar for event in data] #decay times considered from data
	
	signal_region = [event for event in data if range_low <= event.massDiff_d0dstar <= range_up] #events in the signal region
	times_sig = [d.decayTime*1e12 for d in signal_region] #decay times from events in the signal region
	background_region = [event for event in data if event not in signal_region] #events in the background sideband
	times_background = [d.decayTime*1e12 for d in background_region] #decay times from events in the sideband region
	
	N = len(data)
	
	range_tau = np.linspace(0.3, 0.7, 1000) #range of mean lifetime considered for minimisation

	pdf_gaussian_width = .8

	def pdf(ti, l):
		return convoluted_exponential(ti, 1, l, pdf_gaussian_width, 0)

	def negative_log_likelihood(l, ts, mdiffs): #ts, mdiffs are the events' times and mass diffs, l is lifetime
		aaa = [- (1 if range_low <= md <= range_up else wb) * np.log(pdf(x, 1/l)) for x, md in zip(ts, mass_diffs)]
		print(sum(aaa), 1/l)
		return sum(aaa)

	def D_Dtau(tau_x, ts, mdiffs): #first derivative wrt tau
		return derivative(negative_log_likelihood, tau_x, args=(times, mdiffs), dx=1e-5)

	def D_2_Dtau(tau_x, ts, mdiffs):
		return derivative(negative_log_likelihood, tau_x, args=(times, mdiffs), n=2, dx=1e-5)

	# takes tau: initial guess of derivative root
	def Newton_Raphson_tau(tau0): #iterates and finds the root of the derivative (minimum of log likelihood)
		tau_n = tau0
		while True:
			tau_change = D_Dtau(tau_n, times, mass_diffs)/D_2_Dtau(tau_n, times, mass_diffs)
			print('nr', tau_n-tau_change)
			if np.abs(tau_change) <= 1e-4:
				return tau_n-tau_change
			else:
				tau_n -= tau_change


	tau_f = Newton_Raphson_tau(0.4)
	print('tau', tau_f, np.mean(times))
	
	newfig()
	x = np.linspace(0.3, 0.7, 100)
	pl.plot(x, negative_log_likelihood(x, times, mass_diffs))
	savefig('L vs tau')
	
	
	likelyhood = negative_log_likelihood(tau_f, times, mass_diffs)
	
	#statistical uncertainty calculations
	def Neg_Log_1(tau): #Neg Log likelihood shifted by a threshold
		return negative_log_likelihood(tau, times, mass_diffs) - likelyhood - 1 

	def Newton_Raphson_uncertainty(x): #finds the root of Neg_Log_1
		while np.abs(Neg_Log_1(x)) >= 0.005:
			x_n = x - Neg_Log_1(x)/D_Dtau(x, times, mass_diffs)
			x=x_n
		return x
	
	x_1 = Newton_Raphson_uncertainty(tau_f - 0.02)
	x_2 = Newton_Raphson_uncertainty(tau_f + 0.02)
	S = np.abs(x_1 - x_2)/2
	
	print('lifetime ', tau_f, '+- ', S, ' ps')
	return tau_f, S

