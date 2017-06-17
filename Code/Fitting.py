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

# 	np.save('fitting_FULLSET.npy', full_set)
# 	np.save('fitting_AFTERPO.npy', after_po)
# 	np.save('fitting_WIDTH.npy', [deltamass_peak_width])
	
	range_low, range_up = get_sig_range(after_po, deltamass_peak_width)
	print(range_low, range_up)
	
	data = [event for event in full_set if 10e-12 >= event.decayTime]

	wb = calculate_weight(after_po, data, range_low, range_up)
	
	times = [d.decayTime*1e12 for d in data] #decay times considered from data
	mass_diffs = [event.massDiff_d0dstar for event in data] #decay times considered from data
	
	signal_region = [event for event in data if range_low <= event.massDiff_d0dstar <= range_up] #events in the signal region
	times_sig = [d.decayTime*1e12 for d in signal_region] #decay times from events in the signal region
	background_region = [event for event in data if event not in signal_region] #events in the background sideband
	times_background = [d.decayTime*1e12 for d in background_region] #decay times from events in the sideband region
	
	N = len(data)
	
	#Negative log likelihood minimisation:
	range_tau = np.linspace(0.3, 0.7, 1000) #range of mean lifetime considered for minimisation

	def pdf(ti, l):
		return convoluted_exponential(ti, 1, l, 0.8, 0)

	def negative_log_likelihood(l, ts, mdiffs): #ts, mdiffs are the events' times and mass diffs, l is lifetime
		aaa = [- (1 if range_low <= md <= range_up else wb) * np.log(pdf(x, 1/l)) for x, md in zip(ts, mass_diffs)]
		print(sum(aaa)/N, 1/l)
		return sum(aaa)/N

	def D_Dtau(tau_x, ts, mdiffs): #first derivative wrt tau
		return derivative(negative_log_likelihood, tau_x, args=(times, mdiffs), dx=.0001)

	def D_2_Dtau(tau_x, ts, mdiffs):
		return derivative(negative_log_likelihood, tau_x, args=(times, mdiffs), n=2, dx=.0001)

	# takes tau: initial guess of derivative root
	def Newton_Raphson_tau(tau0): #iterates and finds the root of the derivative (minimum of log likelihood)
		tau_n = tau0
		while True:
			tau_change = D_Dtau(tau_n, times, mass_diffs)/D_2_Dtau(tau_n, times, mass_diffs)
			print('nr', tau_n-tau_change)
			if np.abs(tau_change) <= 0.005:
				return tau_n-tau_change
			else:
				tau_n -= tau_change


	tau_f = Newton_Raphson_tau(0.4)
	print('tau', tau_f, np.mean(times))
	return tau_f
