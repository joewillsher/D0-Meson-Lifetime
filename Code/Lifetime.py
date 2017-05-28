from typing import Text
import csv
import numpy as np
from collections import namedtuple
from scipy.constants import c, hbar, physical_constants
import pylab as pl
import scipy.optimize as spo
import lazy_property
from style import *
import os
import sys
from scipy.stats import sem
import scipy.integrate as spi
import scipy.special as sse
cwd = os.getcwd()

# constants
e = physical_constants['electron volt'][0]
m_pi, m_k = 139.57018, 493.677 # TODO: uncert 0.00035, 0.013 respectively

# position 3-vector, units mm
Position = np.array
# momentum 3-vector, units MeV/c
Momentum = np.array
# Momentum = namedtuple('Momentum', ['x', 'y', 'z'])

def magnitude(v: Position):
	return np.sqrt(np.sum(v**2))

def normed(v: Position):
	return v/np.sqrt(np.sum(v**2))

def dot(a: Position, b: Position):
	return np.sum(a*b)

def momentum_toSI(p: float):
	return p * 1e6 * e / c # in SI

def mass_toSI(p: float):
	return p * 1e6 * e / c**2 # in SI

def energy_toSI(p: float):
	return p * 1e6 * e # in SI

def momentum_toMeV(p: float):
	return p * 1e-6 * c / e

def mass_toMeV(p: float):
	return p * 1e-6 * c**2 / e

def energy_toMeV(p: float):
	return p * 1e-6 / e


# Keep all data about the type together in a type, so vals for an event are stored next to each other on the heap.
class CandidateEvent(object):

	def __init__(self, interaction: Position, dstarDecay: Position, d0Decay: Position, kp: Momentum, pd: Momentum, ps: Momentum):
		self.interaction = interaction
		self.dstarDecay = dstarDecay
		self.d0Decay = d0Decay
		self.kp = kp
		self.pd = pd
		self.ps = ps

	def __repr__(self):
		str = super(CandidateEvent, self).__repr__() + "\n"
		str += "Dstar_OWNPV\t" + np.array_str(self.interaction) + "\n"
		str += "Dstar_ENDVERTEX\t" + np.array_str(self.dstarDecay) + "\n"
		str += "D_ENDVERTEX\t" + np.array_str(self.d0Decay) + "\n"
		str += "K\t\t" + np.array_str(self.kp) + "\n"
		str += "Pd\t\t" + np.array_str(self.pd) + "\n"
		str += "Ps\t\t" + np.array_str(self.ps) + "\n"
		return str

	@lazy_property.LazyProperty
	def labFrameTravel(self):
		return magnitude(self.dstarDecay-self.d0Decay) * 1e-3 # convert mm -> m

	@lazy_property.LazyProperty
	def pD0_t(self):
		pcomps_d0 = self.kp+self.pd # in MeV
		return magnitude(pcomps_d0[0:1])

	@lazy_property.LazyProperty
	def pDstar_t(self):
		pcomps_d0 = self.kp+self.pd+self.ps # in MeV
		return magnitude(pcomps_d0[0:1])
	
	@lazy_property.LazyProperty
	def pPslow(self):
		return magnitude(self.ps)

	@lazy_property.LazyProperty
	def pPslow_t(self):
		return magnitude(self.ps[0:1])
	
	@lazy_property.LazyProperty
	def pD0(self):
		pcomps_d0 = self.kp+self.pd # in MeV
		return momentum_toSI(magnitude(pcomps_d0))

	@lazy_property.LazyProperty
	def pDstar(self):
		pcomps_dstar = self.kp+self.pd+self.ps # in MeV
		return momentum_toSI(magnitude(pcomps_dstar))


	def get_daughterEnergy(self, m_pi_si, m_k_si):
		p_pi = momentum_toSI(magnitude(self.pd))
		p_k = momentum_toSI(magnitude(self.kp))
		return np.sqrt((p_pi*c)**2 + mass_toSI(m_pi_si)**2 * c**4)   +   np.sqrt((p_k*c)**2 + mass_toSI(m_k_si)**2 * c**4)

	@lazy_property.LazyProperty
	def daughterEnergy(self):
		return self.get_daughterEnergy(m_pi, m_k)

	@lazy_property.LazyProperty
	def daughterEnergy_pp(self):
		return self.get_daughterEnergy(m_pi, m_pi)

	@lazy_property.LazyProperty
	def daughterEnergy_kk(self):
		return self.get_daughterEnergy(m_k, m_k)

	
	def get_reconstructedMass(self, daughterEnergy):
		return np.sqrt(daughterEnergy**2 - (self.pD0*c)**2)/c**2

	@lazy_property.LazyProperty
	def reconstructedD0Mass(self):
		return self.get_reconstructedMass(self.daughterEnergy)

	@lazy_property.LazyProperty
	def reconstructedD0Mass_pp(self):
		return self.get_reconstructedMass(self.daughterEnergy_pp)

	@lazy_property.LazyProperty
	def reconstructedD0Mass_kk(self):
		return self.get_reconstructedMass(self.daughterEnergy_kk)



	def get_dStarEnergy(self, daughterEnergy):
		p_pislow = momentum_toSI(magnitude(self.ps))
		m_pi_si = mass_toSI(m_pi)
		return daughterEnergy + np.sqrt((p_pislow*c)**2 + m_pi_si**2 * c**4)

	def get_reconstructedDstarMass(self, starEnergy):
		return np.sqrt(starEnergy**2 - magnitude(self.pDstar*c)**2)/c**2


	@lazy_property.LazyProperty
	def dStarEnergy(self):
		return self.get_dStarEnergy(self.daughterEnergy)

	@lazy_property.LazyProperty
	def reconstructedDstarMass(self):
		return self.get_reconstructedDstarMass(self.dStarEnergy)

	# in MeV
	@lazy_property.LazyProperty
	def massDiff_d0dstar(self):
		return mass_toMeV(self.reconstructedDstarMass - self.reconstructedD0Mass)

	@lazy_property.LazyProperty
	def dStarEnergy_kk(self):
		return self.get_dStarEnergy(self.daughterEnergy_kk)

	@lazy_property.LazyProperty
	def reconstructedDstarMass_kk(self):
		return self.get_reconstructedDstarMass(self.dStarEnergy_kk)

	# in MeV
	@lazy_property.LazyProperty
	def massDiff_d0dstar_kk(self):
		return mass_toMeV(self.reconstructedDstarMass_kk - self.reconstructedD0Mass_kk)

	@lazy_property.LazyProperty
	def dStarEnergy_pp(self):
		return self.get_dStarEnergy(self.daughterEnergy_pp)

	@lazy_property.LazyProperty
	def reconstructedDstarMass_pp(self):
		return self.get_reconstructedDstarMass(self.dStarEnergy_pp)

	# in MeV
	@lazy_property.LazyProperty
	def massDiff_d0dstar_pp(self):
		return mass_toMeV(self.reconstructedDstarMass_pp - self.reconstructedD0Mass_pp)




	@lazy_property.LazyProperty
	def gamma(self):
		p_d0 = self.pD0
		m_d0 = self.reconstructedD0Mass
		return p_d0 / (c * m_d0)

	@lazy_property.LazyProperty
	def decayTime(self):
		x = self.labFrameTravel
		m_d0 = self.reconstructedD0Mass
		p_d0 = self.pD0
		return x * m_d0 / p_d0


	@lazy_property.LazyProperty
	def d0IP_log(self):
		x = (self.d0Decay - self.dstarDecay)*1e-3
		p = normed(momentum_toSI(self.kp+self.pd))
		return np.log10(magnitude(x - dot(x,p) * p)*1e3)
	
	@lazy_property.LazyProperty
	def kIP_log(self):
		x = (self.d0Decay - self.dstarDecay)*1e-3
		p = normed(momentum_toSI(self.kp))
		return np.log10(magnitude(x - dot(x,p) * p)*1e3)

	@lazy_property.LazyProperty
	def pIP_log(self):
		x = (self.d0Decay - self.dstarDecay)*1e-3
		p = normed(momentum_toSI(self.pd))
		return np.log10(magnitude(x - dot(x,p) * p)*1e3)

	@lazy_property.LazyProperty
	def psIP_log(self):
		x = (self.d0Decay - self.dstarDecay)*1e-3
		p = normed(momentum_toSI(self.ps))
		return np.log10(magnitude(x - dot(x,p) * p)*1e3)

	@lazy_property.LazyProperty
	def pk_t(self):
		return magnitude(self.kp[0:1])

	@lazy_property.LazyProperty
	def pp_t(self):
		return magnitude(self.pd[0:1])

	@lazy_property.LazyProperty
	def s_z(self):
		return self.dstarDecay[2] - self.interaction[2]

	@lazy_property.LazyProperty
	def costheta(self):
		p, r = self.kp+self.pd+self.ps, self.dstarDecay-self.interaction
		return dot(p/magnitude(p), r/magnitude(r))
		
		






# reads a file and returns the D0 candidate events it lists
# - expects the file to have specific col titles, returns None if there is an error
def readFile(name: Text):
    with open(cwd+'/'+name, 'r') as csvfile:
    	# get the rows from the file
        rows = csv.reader(csvfile, delimiter=' ', quotechar='|') #creates array of array: each array is a row.

        # take the first element of the generator as the header
        header = next(rows)
        print(header)
        # store the candidates here
        cands = []

		# ignore the coordinate, remove the last '_X'/'_PX' part in the header name
        header_raw_names = ['_'.join(name.split('_')[:-1]) for name in header[::3]]
        order = ['Dstar_OWNPV', 'Dstar_ENDVERTEX', 'D_ENDVERTEX', 'K', 'Pd', 'Ps']
        print('names', header_raw_names)
        # get the indicies of these params in the row
        elementIdxs = [header_raw_names.index(x) for x in order]
        print(elementIdxs)

        for row in rows:
        	# reshape row into several 3-vectors
        	nums = [float(x) for x in row]
        	data = np.reshape(np.array(nums), (len(nums)//3, int(3)))
        	cand = CandidateEvent(data[elementIdxs[0]], data[elementIdxs[1]], data[elementIdxs[2]], data[elementIdxs[3]], data[elementIdxs[4]], data[elementIdxs[5]])
        	cands.append(cand)

        return cands





def convoluted_exponential(t, A, l, s, m):
	if s < 0 and m == 0:
		return 0
	return A* l/2 * np.exp(2*m + l * s**2 - 2*t) * sse.erfc((m + l * s**2 - t)/(2**0.5 * s))






def plotData(data):
	# mass dist
	masses = [mass_toMeV(d.reconstructedD0Mass) for d in data]
	newfig()
	pl.hist(masses, bins=100, histtype='step', fill=False)
	pl.xlabel(r'$D^0$ Mass [MeV/$c^2$]')
	savefig('mass-dist')
	pl.close()

	# dstar mass dist
	ds_masses = [mass_toMeV(d.reconstructedDstarMass) for d in data]
	newfig()
	pl.hist(ds_masses, bins=100, histtype='step', fill=False)
	pl.xlabel(r'$D^{+*}$ Mass [MeV/$c^2$]')
	savefig('dstar-mass-dist')
	pl.close()

	# mass difference dist
	mass_diffs = [x0 - x1 for x0, x1 in zip(ds_masses, masses)]
	newfig()
	pl.hist(mass_diffs, bins=100, histtype='step', fill=False)
	pl.xlabel(r'Mass difference [MeV/$c^2$]')
	savefig('mass-diff-dist')
	pl.close()

	# gamma dist
	gammas = [d.gamma for d in data]
	newfig()
	pl.hist(gammas, bins=500)
	savefig('gamma-dist')
	pl.close()

	# travel dist
	trav = [d.labFrameTravel for d in data]
	newfig()
	pl.hist(trav, bins=500, range=(0, 0.04))
	savefig('trav-dist')
	pl.close()


def calculateLifetime(data, bg, bg_fraction):
	# decay time dist
	times = [d.decayTime*1e12 for d in data if d.decayTime < 10e-12]
	bg_times = [d.decayTime*1e12 for d in bg if d.decayTime < 10e-12]
	np.save('TIMES', times)
	np.save('BG_TIMES', bg_times)
	
	newfig()
	pl.hist(times, bins=100, range=(0, 50))
	savefig('time-hist')
	pl.close()

	# decay time curve
	hist, bin_edges = np.histogram(times, bins=120, range=(0., 10.))
	bg_hist, bg_bin_edges = np.histogram(bg_times, bins=120, range=(0., 10.))

	num_events = np.sum(hist)
	num_bg = np.sum(bg_hist)
	# get the mean value in each bin
	sy, _ = np.histogram(times, bins=bin_edges, weights=times)
	time = np.array([e if n == 0 else t/n for t, n, e in zip(sy, hist, bin_edges[1:])])
	print(time)

	bg_hist_normalised = bg_hist/num_bg * bg_fraction * num_events
	subtracted_hist = hist - bg_hist_normalised
	print(subtracted_hist, np.sum(subtracted_hist))
	errors = [x*.999999 if x <= 1  else np.sqrt(x) for x in subtracted_hist-0.01]

	# decay time fitting
	po, po_cov = spo.curve_fit(lambda t, A, tau: A * np.exp(-t/tau), time, subtracted_hist, [num_events, 1.5])

	po_conv, po_cov_conv = spo.curve_fit(convoluted_exponential, time, subtracted_hist, [num_events/2, .41, .05, 0.], errors, absolute_sigma=True)

	newfig()
	pl.semilogy(time, hist, '.g')
	pl.semilogy(time, subtracted_hist, '.r')
	pl.errorbar(time, subtracted_hist, yerr=errors, fmt=',r', capsize=0)
# 	if not is_latex:
# 		pl.semilogy(time, np.vectorize(lambda t: po[0] * np.exp(-t/po[1]))(time), '-g')
# 		pl.semilogy(time, np.vectorize(lambda t: po[0] * np.exp(-t/np.mean(times)))(time), '-g')
	pl.semilogy(time, convoluted_exponential(time, *po_conv), '-b')
	pl.xlabel(r'Decay time [ps]')
	savefig('decay-fitted')
	pl.close()

	newfig()
	pl.plot(time, hist, '-g')
	pl.plot(time, subtracted_hist, '-r')
	pl.errorbar(time, subtracted_hist, yerr=errors, fmt=',r', capsize=0)
	# 	if not is_latex:
	# 		pl.plot(time, np.vectorize(lambda t: po[0] * np.exp(-t/po[1]))(time), '-g')
	# 		pl.plot(time, np.vectorize(lambda t: po[0] * np.exp(-t/np.mean(times)))(time), '-g')
	pl.plot(time, convoluted_exponential(time, *po_conv), '-b')
	pl.xlabel(r'Decay time [ps]')
	savefig('decay')
	pl.close()


	partial_lifetime = po[1]
	mean_lifetime = np.mean(times)
	print('convpo=', po_conv, '+-', np.sqrt(po_cov_conv[1][1]))
	print('partial lifetime\t' + str(partial_lifetime) + ' ps', 'OR MEAN PL =', str(mean_lifetime)+'ps', 'OR CONV=', str(po_conv[1])+'ps')
	
	with open("data.txt", "w") as text_file:
	    text_file.write("lifetime=%s\n" % np.round(mean_lifetime*1e3, 0))
	    text_file.write("lifetime_conv=%s\n" % np.round(po_conv[1]*1e3, 0))
	    text_file.write("lifetime_exp=%s\n" % np.round(partial_lifetime*1e3, 0))







def plot_compare(accepted, rejected, prop, name, range=None, label=None):
	diffs_a, diffs_r = [getattr(d, prop) for d in accepted], [getattr(d, prop) for d in rejected]
	fig, ax = newfig()
	pl.yscale('log')
	
	acc = pl.hist(diffs_a, 100, facecolor='g', histtype='step', range=range, label='accepted')
	rej = pl.hist(diffs_r, 100, facecolor='r', histtype='step', range=range, label='rejected')
	
	hist_r, bin_edges_r = np.histogram(diffs_r, bins=100, range=range, normed=True)
	hist_a, bin_edges_a = np.histogram(diffs_a, bins=100, range=range, normed=True)
	
	if label:
		pl.xlabel(label)
	pl.ylabel(r'Relative frequency')
	ax.legend(loc='upper right', shadow=False)
	savefig(name+'-compare')
	pl.close()





def background_fit(dm, bg_A, bg_p):
	return bg_A * (dm-m_pi)**bg_p
	
def signal_fit(dm, sig_A, sig_centre, sig_w):
	return sig_A * np.exp(-(dm-sig_centre)**2/(2*sig_w**2))
	
def combined_fit(dm, bg_A, bg_p, sig_A, sig_centre, sig_w):
	return signal_fit(dm, sig_A, sig_centre, sig_w) + background_fit(dm, bg_A, bg_p)








def massDiff_plot(events, ext_name='', expected_bg=30, range=(139, 165), methodName='massDiff_d0dstar'):
	initial = [expected_bg, 0.25, 90, 146, .65]

	diffs = [getattr(d, methodName) for d in events if getattr(d, methodName) < 165]
	hist, bin_edges = np.histogram(diffs, bins=100)
	N = np.sum(diffs)
	masses = np.array([np.mean([d0, d1]) for d0, d1 in zip(bin_edges[:-1], bin_edges[1:])])
	bin_width = bin_edges[1] - bin_edges[0]

	# https://suchideas.com/articles/maths/applied/histogram-errors/
	errors = np.sqrt(hist)
			
	po, po_cov = spo.curve_fit(combined_fit, masses, hist, initial, sigma=errors)
	
	print('po-fit', po)
	fig, ax = newfig()
	pl.plot(masses, hist, '.r')
	pl.errorbar(masses, hist, yerr=errors, fmt=',r', capsize=0)
	
	masses_continuous = np.arange(m_pi, masses[-1], .1)
	pl.plot(masses_continuous, signal_fit(masses_continuous, *po[2:]), '-g')
	pl.fill_between(masses_continuous, 0, background_fit(masses_continuous, *po[:2]), facecolor='blue', edgecolor="None", alpha=0.5)
	ax.set_xlim(range)
	pl.xlabel(r'$\Delta m$ [GeV/$c^2$]')
	pl.ylabel(r'Relative frequency')
	savefig('cut-fitted'+ext_name)
	pl.close()
	return po, bin_width






# takes list of candidate events, cuts them by their mass diff
def cutEventSet_massDiff(events, width):
	po, bin_width = massDiff_plot(events)

	# cut at 4 widths
	bg_A, bg_p, sig_A, sig_centre, sig_w = po
	range_low, range_up = sig_centre - sig_w*width, sig_centre + sig_w*width
	print('range', range_low, range_up)
	accepted = [event for event in events if range_low <= event.massDiff_d0dstar <= range_up]
	rejected = [event for event in events if not range_low <= event.massDiff_d0dstar <= range_up]
	return accepted, rejected, po, bin_width





def cut(accepted, rejected, cond):
	acc, rej = [], list(rejected)
	for a in accepted:
		if cond(a):
			acc.append(a)
		else:
			rej.append(a)

	return np.array(acc), np.array(rej)




# def error_hisogram(x, xbins, y):
# 	n, _ = np.histogram(x, bins=xbins)
# 	sy, _ = np.histogram(x, bins=xbins, weights=y)
# 	sy2, _ = np.histogram(x, bins=xbins, weights=y*y)
# 	mean = sy / n
# 	return np.sqrt(sy2/n - mean*mean)/np.sqrt(n)




def estimate_background(po, filtered, width):
	sig_centre, sig_w = po[3], po[4]
	range_low, range_up = sig_centre - sig_w*width, sig_centre + sig_w*width
	bg_integral = spi.quad(background_fit, range_low, range_up, args=(po[0], po[1]))[0]

	bg_fraction = bg_integral/len(filtered)

	print("BACKGROUND EST", bg_integral, len(filtered), bg_fraction)
	return bg_integral, bg_fraction
