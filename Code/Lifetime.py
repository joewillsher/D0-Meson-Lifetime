from typing import Text
import csv
import numpy as np
from collections import namedtuple
import pylab as pl
import scipy.optimize as spo
import lazy_property
from style import *
import os
import sys
from scipy.stats import sem
from Fitting import maximum_likelyhood_exp_fit
from Background import *
cwd = os.getcwd()


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

	def __init__(self, interaction: Position, dstarDecay: Position, d0Decay: Position, bDecay: Position, kp: Momentum, pd: Momentum, ps: Momentum):
		self.interaction = interaction
		self.bDecay = bDecay
		self.dstarDecay = dstarDecay
		self.d0Decay = d0Decay
		self.kp = kp
		self.pd = pd
		self.ps = ps

	def __repr__(self):
		str = super(CandidateEvent, self).__repr__() + "\n"
		str += "Dstar_OWNPV\t" + np.array_str(self.interaction) + "\n"
		str += "B_ENDVERTEX\t" + np.array_str(self.bDecay) + "\n"
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
	def dStarlabFrameTravel(self):
		return magnitude(self.bDecay-self.dstarDecay) * 1e-3 # convert mm -> m

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

	@lazy_property.LazyProperty
	def pDstar_comps(self):
		pcomps_dstar = self.kp+self.pd+self.ps # in MeV
		return momentum_toSI(pcomps_dstar)

	@lazy_property.LazyProperty
	def pD0_comps(self):
		pcomps_dstar = self.kp+self.pd # in MeV
		return momentum_toSI(pcomps_dstar)


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
		x = (self.d0Decay-self.dstarDecay)*1e-3
		m_d0 = self.reconstructedD0Mass
		p_d0 = self.pD0_comps
		return np.sign(dot(x, p_d0)) * magnitude(x) * m_d0 / magnitude(p_d0)

	@lazy_property.LazyProperty
	def dStarDecayTime(self):
		x = (self.dstarDecay-self.bDecay)*1e-3
		m_ds = self.reconstructedDstarMass
		p_ds = self.pDstar_comps
		return np.sign(dot(x, p_ds)) * magnitude(x) * m_ds / magnitude(p_ds)


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
        order = ['Dstar_OWNPV', 'Dstar_ENDVERTEX', 'D_ENDVERTEX', 'B_ENDVERTEX', 'K', 'Pd', 'Ps']
        print('names', header_raw_names)
        # get the indicies of these params in the row
        elementIdxs = [header_raw_names.index(x) for x in order]
        print(elementIdxs)

        for row in rows:
        	# reshape row into several 3-vectors
        	nums = [float(x) for x in row[:-1]] # remove Dstar_FD ([:-1]) bc scalars don't work
        	data = np.reshape(np.array(nums), (len(nums)//3, int(3)))
        	cand = CandidateEvent(data[elementIdxs[0]], data[elementIdxs[1]], data[elementIdxs[2]], data[elementIdxs[3]], data[elementIdxs[4]], data[elementIdxs[5]], data[elementIdxs[6]])
        	cands.append(cand)

        return cands


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


def calculateLifetime(data, bg, deltamass_po, deltamass_peak_width):
	bg_integral, sig_integral, bg_fraction = estimate_background(deltamass_po, data, deltamass_peak_width)
	
	tau_elimination, tau_elimination_err, wb = maximum_likelyhood_exp_fit(data, deltamass_po, deltamass_peak_width)
	
	filtered = [d for d in data if 0 <= d.decayTime < 10e-12]

	range_low, range_up = 142.414041989, 149.059198523

	times = [d.decayTime*1e12 for d in filtered]
	np.save('TIMES', times)
	weights = [(1 if range_low <= d.massDiff_d0dstar <= range_up else wb) for d in filtered]

	time_range, bin_num = (0, 10), 120

	# decay time curve
	hist, bin_edges = np.histogram(times, bins=bin_num, range=time_range, weights=weights)	
	hist_bg = np.histogram(times, bins=bin_num, range=time_range, normed=True)[0] * np.sum(hist) * bg_fraction

	sy = np.histogram(times, bins=bin_edges, weights=times)[0]
	time = np.array([e if n == 0 else t/n for t, n, e in zip(sy, hist, bin_edges[1:])])
	errors = [x*.9999999999 if x <= 1 else np.sqrt(x) for x in hist-.0000000001]

	newfig()
	pl.semilogy(time, hist, '.g')
	pl.semilogy(time, hist_bg, '.k')
	pl.errorbar(time, hist, yerr=errors, fmt=',r', capsize=0)
	# pl.semilogy(time, convoluted_exponential(time, *po_conv), '-b')
	pl.xlabel(r'Decay time [ps]')
	savefig('decay-fitted')
	pl.close()

# 	po_conv, po_cov_conv = spo.curve_fit(convoluted_exponential, time, hist, [num_events/2, .41, .8, 0.], errors, absolute_sigma=True)
# 
# 	partial_lifetime = po[1]
# 	mean_lifetime = np.mean(times)
# 	print('convpo=', po_conv, '+-', np.sqrt(po_cov_conv[1][1]))
# 	print('partial lifetime\t' + str(partial_lifetime) + ' ps', 'OR MEAN PL =', str(mean_lifetime)+'ps', 'OR CONV=', str(1/po_conv[1])+'ps')
	
# 	newfig()
# 	pl.semilogy(time, hist, '.g')
# 	pl.semilogy(time, subtracted_hist, '.r')
# 	pl.errorbar(time, subtracted_hist, yerr=errors, fmt=',r', capsize=0)
# # 	if not is_latex:
# # 		pl.semilogy(time, np.vectorize(lambda t: po[0] * np.exp(-t/po[1]))(time), '-g')
# # 		pl.semilogy(time, np.vectorize(lambda t: po[0] * np.exp(-t/np.mean(times)))(time), '-g')
# 	pl.semilogy(time, convoluted_exponential(time, *po_conv), '-b')
# 	pl.xlabel(r'Decay time [ps]')
# 	savefig('decay-fitted')
# 	pl.close()
# 
# 	newfig()
# 	pl.plot(time, subtracted_hist, ',r')
# 	pl.errorbar(time, subtracted_hist, yerr=errors, fmt=',r')
# 	pl.plot(time, hist, '.r')
# # 	if not is_latex:
# # 		pl.plot(time, np.vectorize(lambda t: po[0] * np.exp(-t/po[1]))(time), '-g')
# # 		pl.plot(time, np.vectorize(lambda t: po[0] * np.exp(-t/np.mean(times)))(time), '-g')
# 	pl.plot(time, convoluted_exponential(time, *po_conv), '-b')
# 	pl.xlabel(r'Decay time [ps]')
# 	savefig('decay')	
# 	pl.close()
# 
# 	newfig()
# 	pl.plot(time, subtracted_hist, ',r')
# 	pl.errorbar(time, subtracted_hist, yerr=errors, fmt=',r')
# 	pl.plot(time, hist, '.r')
# 	pl.plot(time, convoluted_exponential(time, *[po_conv[0], tau_elimination, .8, 0.]), '-b')
# 	pl.xlabel(r'Decay time [ps]')
# 	savefig('decay-BACKGROUNDSIUBTR')
# 	pl.close()

	
	with open("data.txt", "w") as text_file:
# 	    text_file.write("lifetime=%s\n" % np.round(mean_lifetime*1e3, 1))
# 	    text_file.write("lifetime_conv=%s\n" % np.round(1/po_conv[1]*1e3, 1))
# 	    text_file.write("lifetime_exp=%s\n" % np.round(partial_lifetime*1e3, 1))
	    text_file.write("lifetime_bgreduction=%s\n" % np.round(tau_elimination*1e3, 1))
	    text_file.write("error_bgreduction=%s\n" % np.round(tau_elimination_err*1e3, 1))







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












def massDiff_plot(events, ext_name='', fit=True, bg_ratio=0.15, range=(139, 165), methodName='massDiff_d0dstar'):
	diffs = [getattr(d, methodName) for d in events if getattr(d, methodName) < 165]
	hist, bin_edges = np.histogram(diffs, bins=100)
	N = np.sum(diffs)
	sy, _ = np.histogram(diffs, bins=bin_edges, weights=diffs)
	masses = np.array([e if n == 0 else t/n for t, n, e in zip(sy, hist, bin_edges[1:])])
	bin_width = bin_edges[1] - bin_edges[0]

	initial = [max(hist)*0.2666666667*bg_ratio, 0.25, m_pi, max(hist)*1.2, 145.5, .1, 1., .9]

	# https://suchideas.com/articles/maths/applied/histogram-errors/
	errors = np.sqrt(hist)
	x_errors = np.repeat(bin_width/2, len(errors))
	
	if fit:	
		po, po_cov = spo.curve_fit(combined_fit, masses, hist, initial, sigma=errors, bounds=([0, .25, 139, 0, 0, 0, 0, 0], [np.inf, .33, 140, np.inf, np.inf, 1, 5, np.inf]))
		print('po-fit', po)
	
	fig = newrawfig(width=2)
	margin = .1
	out_margin = .02
	subpl_height = .25
	width, height = 1, 1
	# x_l, x_b, w, h
	ax = fig.add_axes([margin, subpl_height, width-margin-out_margin, height-subpl_height-margin])
	ax.axes.get_xaxis().set_visible(False)
	ax.plot(masses, hist, '.r')
	ax.errorbar(masses, hist, yerr=errors, fmt=',r', capsize=0)
	
	if fit:
		masses_continuous = np.arange(m_pi, masses[-1], .02)
		ax.plot(masses_continuous, combined_fit(masses_continuous, *po), '-g')
# 		ax.plot(masses_continuous, gaussian(masses_continuous, po[2] * po[7], po[5], po[4]), '-k')
# 		ax.plot(masses_continuous, gaussian(masses_continuous, po[2] * (1-po[7]), po[6], po[4]), '-k')
		ax.fill_between(masses_continuous, 0, background_fit(masses_continuous, *po[:3]), facecolor='blue', edgecolor="None", alpha=0.35)
		
		pull_ax = fig.add_axes([margin, margin, width-margin-out_margin, subpl_height-margin])
		pulls = (hist-combined_fit(masses, *po))/errors
		print(pulls)
		pull_ax.bar(masses, pulls, bin_width, edgecolor="None")
		pull_ax.set_xlim(range)
# 		pull_ax.set_ylim(-5,5)
			
		pull_ax.set_ylabel(r'Pull')
		pull_ax.set_xlabel(r'$\Delta m$ [GeV/$c^2$]')
		
		for tick in pull_ax.yaxis.get_major_ticks():
			tick.label.set_fontsize(6)

		fig.set_tight_layout(True)
	else:
		ax.set_xlabel(r'$\Delta m$ [GeV/$c^2$]')
	
	ax.set_xlim(range)
	ax.set_ylabel(r'Relative frequency')
	savefig('cut-fitted'+ext_name)
	pl.close()
	return po if fit else [], bin_width



# takes list of candidate events, cuts them by their mass diff
def cutEventSet_massDiff(events, width):
	po, bin_width = massDiff_plot(events)

	# cut at 4 widths
	range_low, range_up = get_sig_range(po, width)
	print('range', range_low, range_up)
	accepted = [event for event in events if range_low <= event.massDiff_d0dstar <= range_up]
	rejected = [event for event in events if not range_low <= event.massDiff_d0dstar <= range_up]
	return accepted, rejected, po, bin_width
