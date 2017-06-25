from style import *

range=(139, 165)

fig = newrawfig(width=.65)
margin = .2
out_margin = .02
subpl_height = .35
width, height = 1, 1
# x_l, x_b, w, h
ax = fig.add_axes([margin, subpl_height, width-margin-out_margin, height-subpl_height-margin/2])
ax.axes.get_xaxis().set_visible(False)

ax.plot(0, 100000)

if True:

	pull_ax = fig.add_axes([margin, margin, width-margin-out_margin, subpl_height-margin])
	# pull_ax.set_xlim(range)
# 		pull_ax.set_ylim(-5,5)

	pull_ax.set_ylabel(r'Pull')
	pull_ax.set_xlabel(r'$\Delta m$ [GeV/$c^2$]')

	for tick in pull_ax.yaxis.get_major_ticks():
		tick.label.set_fontsize(4 if is_latex else 6)

	fig.set_tight_layout(True)
else:
	ax.set_xlabel(r'$\Delta m$ [GeV/$c^2$]')

ax.set_xlim(range)
ax.set_ylabel(r'Relative frequency')
savefig('cut-fitted')
pl.close()
