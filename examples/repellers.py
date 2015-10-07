"""
Example of pybo run on the repellers simulator.
"""
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import animation

from ezplot import figure, show
import pybo
from benchfunk.functions import RepellersModel

ninit = 10
niter = 100
niter_per_frame = 1

f = RepellersModel(
    weights=[1, 2, 2, 3],
    centers=[( 2, -7),
             (-3, -3),
             ( 3, -3),
             (-6, -4)],
    scales=[(0.5, 0.5),
            (0.5, 0.5),
            (0.5, 0.5),
            (0.5, 0.5)],
    npaths=100)				# average reward over npaths

bounds = np.array([
    [0., .25], [-8, 8], [-8, 0],
    [0., .25], [-8, 8], [-8, 0],
    [0., .25], [-8, 8], [-8, 0]
])

# initialize model
model = pybo.init_model(f, bounds, ninit=ninit)
policies = ['EI', 'PI', 'UCB', 'Thompson']
npols = len(policies)
models = [model.copy() for _ in policies]
ybests = np.zeros((npols, niter))

# initialize figure
fig = figure(figsize=(4*npols, 4))
axs = [fig.add_subplotspec((2, npols), (0, i), aspect='equal')
	   for i in xrange(npols)]
axs += [fig.add_subplotspec((2, npols), (1, 0), colspan=npols)]

def animate(frame):
	tlast = (frame + 1) * niter_per_frame

	for i, policy in enumerate(policies):
		policy = getattr(pybo.policies, policy)

		for t in xrange(frame*niter_per_frame, tlast):
			# get selection
			index = policy(models[i], bounds, models[i]._models[0]._X)
			x, _ = pybo.solvers.solve_lbfgs(index, bounds)

			# query black-box and update model
			y = f(x)
			models[i].add_data(x, y)

			# get two different recommendations to compare
			xbest = pybo.recommenders.best_incumbent(
				models[i],
				bounds,
				models[i]._models[0]._X)
			ybests[i, t] = f(xbest)

		# render frame
		f.plot(axs[i], xbest, N=50, horizon=100)
		axs[i].set_title(policies[i], fontsize=14)
		axs[i].axis('off')

	axs[-1].cla()
	axs[-1].plot(xrange(ninit+1, ninit+tlast+1), ybests[:, :tlast].T)
	axs[-1].legend(policies, loc=4)
	axs[-1].set_xlim([1, ninit+niter])
	axs[-1].set_xlabel('Function evaluations', fontsize=14)
	axs[-1].set_ylabel('Incumbent', fontsize=14)

# get animation for niter frames
anim = animation.FuncAnimation(fig,
                               animate,
                               frames=niter // niter_per_frame,
                               interval=10)
plt.show()
# save gif of animation
# anim.save('repellers.gif', writer='imagemagick', fps=2)
