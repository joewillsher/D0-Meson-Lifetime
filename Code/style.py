import matplotlib as plt
import numpy as np
import sys
import matplotlib.pyplot as pl

# http://bkanuka.com/articles/native-latex-plots/

golden_mean = (np.sqrt(5.0)-1.0)/2.0            # Aesthetic ratio (you could change this)

fig_width_pt = 370                        # Get this from LaTeX using \the\textwidth
inches_per_pt = 1.0/72.27                       # Convert pt to inch

def figsize(scale, height_adj=0):
	fig_width = fig_width_pt*inches_per_pt*scale    # width in inches
	fig_height = fig_width*golden_mean + height_adj              # height in inches
	fig_size = [fig_width, fig_height]
	return fig_size

is_latex = '--latex-plot' in sys.argv
if is_latex:
	plt.use('pgf')
	pgf_with_latex = {                      # setup matplotlib to use latex for output
	"pgf.texsystem": "pdflatex",        # change this if using xetex or lautex
	"text.usetex": True,                # use LaTeX to write all text
	"font.family": "serif",
	"font.serif": [],                   # blank entries should cause plots to inherit fonts from the document
	"lines.markersize": 2,
	"lines.linewidth" : .4,
	"boxplot.boxprops.linewidth": .4,
	"font.sans-serif": [],
	"font.monospace": [],
	"axes.labelsize": 8,               # LaTeX default is 10pt font.
	"font.size": 8,
	"legend.fontsize": 7,               # Make the legend/label fonts a little smaller
	"xtick.labelsize": 7,
	"ytick.labelsize": 7,
	"figure.figsize": figsize(0.7),     # default fig size of 0.7 textwidth
	"figure.autolayout" : True,
	"pgf.preamble": [
		r"\usepackage[utf8x]{inputenc}",    # use utf8 fonts becasue your computer can handle it :)
		r"\usepackage[T1]{fontenc}",        # plots will be generated using this preamble
		r"\usepackage{/Users/joe/Documents/SummerProject/Report/atlasphysics}",    # use utf8 fonts because your computer can handle it :)
		]
	}
	plt.rcParams.update(pgf_with_latex)

	# I make my own newfig and savefig functions

	default_width = .7

	def newrawfig(width=.7, height_adj=0):
		pl.clf()
		fig = pl.figure(figsize=figsize(width, height_adj))
		return fig

	def newfig(width=.7, height_adj=0):
		pl.clf()
		fig = pl.figure(figsize=figsize(width, height_adj))
		ax = fig.add_subplot(111)
		return fig, ax

	def savefig(filename, directory=''):
		pl.savefig(directory+filename+'.pgf')
		pl.savefig(directory+filename+'.png')

	def savefig_image(filename, directory=''):
		pl.savefig(directory+filename+'.pgf', dpi=4000)
		pl.savefig(directory+filename+'.png')
else:

	default_width = 2

	def newfig(width=2, height_adj=0):
		pl.clf()
		fig = pl.figure(figsize=figsize(width, height_adj))
		ax = fig.add_subplot(111)
		return fig, ax

	def newrawfig(width=2, height_adj=0):
		pl.clf()
		fig = pl.figure(figsize=figsize(width, height_adj))
		return fig

	def savefig(filename, directory=''):
		pl.savefig(directory+filename+'.png')

	def savefig_image(filename, directory=''):
		savefig(filename, directory)


# # Simple plot
# fig, ax  = newfig(0.6)
# def ema(y, a):
#     s = []
#     s.append(y[0])
#     for t in range(1, len(y)):
#     	s.append(a * y[t] + (1-a) * s[t-1])
#     return np.array(s)
#
# y = [0]*200
# y.extend([20]*(1000-len(y)))
# s = ema(y, 0.01)
# ax.plot(s)
# ax.set_xlabel('X Label')
# ax.set_ylabel(r'omg EMA $y \pi$ \GeV')
# savefig('ema')
