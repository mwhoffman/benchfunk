from collections import OrderedDict
import itertools
from benchfunk import functions
from benchfunk.core import run_stack

niter = 100
nreps = 100

# prescribe policies to try
policies = [('EI', dict(xi=0.0)),
            ('PI', dict(xi=0.1)),
            ('PES', dict(opes=False)),
            ('PES', dict(opes=True))]

# prescribe models to try
models = [None]

# prescribe functions to test on
problems = [functions.Gramacy,
            functions.Branin,
            functions.Goldstein,
            functions.Bohachevsky,
            functions.Hartmann3,
            functions.Hartmann6]

sn2_values = [0.01, 0.05, 0.10]

problems = itertools.product(problems, sn2_values)

# build stack of experiments
stack = OrderedDict()
for problem, sn2 in problems:
    name = '{0}({1:.2f})'.format(problem, sn2)
    stack[name] = dict(problem=(problem, dict(sn2=sn2)))

# for each stack

results = run_stack(stack, nreps)