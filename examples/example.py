"""
Example script which runs a number of policies on a single problem with varying
levels of observation noise.
"""

from collections import OrderedDict
from benchfunk.core import run_stack, plot_stack

from pybo.policies import *
from benchfunk.functions import *

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
policies['EI(0.0)'] = (EI, dict(xi=0.0))
policies['PI(0.1)'] = (PI, dict(xi=0.1))
policies['TS(100)'] = (Thompson, dict(n=100))

# run stack of experiments
results = run_stack(problems, policies, niter, nreps, name=name)
results.name = name

# plot results
fig = plot_stack(results, problems.keys(), policies.keys(), name=name)
fig.name = '.'.join([name, 'plot'])
