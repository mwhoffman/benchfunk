from collections import OrderedDict
import pybo.policies as pp

from benchfunk.classics import *
from benchfunk.runner import run_stack
from benchfunk.plotter import plot_stack


# parameters
name = __name__
niter = 10
nreps = 3

# prescribe problem instances
problems = OrderedDict()
problems['Gramacy(0.01)'] = (Gramacy, dict(sn2=0.01))
problems['Gramacy(0.05)'] = (Gramacy, dict(sn2=0.05))
problems['Gramacy(0.10)'] = (Gramacy, dict(sn2=0.10))

# prescribe policies to try
policies = OrderedDict()
policies['EI(0.0)'] = (pp.EI, dict(xi=0.0))
policies['PI(0.1)'] = (pp.PI, dict(xi=0.1))
policies['TS(100)'] = (pp.Thompson, dict(n=100))

# run stack of experiments
results = run_stack(problems, policies, niter, nreps, name=name)
results.name = name

# plot results
fig = plot_stack(results, problems.keys(), policies.keys(), name=name)
fig.name = '.'.join([name, 'plot'])
