from collections import OrderedDict
import itertools
from benchfunk import functions
from benchfunk.core import run_instance, run_stack

niter = 100
nreps = 10

# prescribe policies to try
policies = [('EI', dict(xi=0.0)),
            ('PI', dict(xi=0.1)),
            ('PES', dict(opes=False)),
            ('PES', dict(opes=True))]

# prescribe models to try
models = [None]

# prescribe functions to test on and experimental setups, in this case simply
# noise levels
funcs = [functions.Gramacy,
         functions.Branin,
         functions.Goldstein,
         functions.Bohachevsky,
         functions.Hartmann3,
         functions.Hartmann6]

setups = [dict(sn2=0.01),
          dict(sn2=0.05),
          dict(sn2=0.10)]

problems = itertools.product(funcs, setups)

# list all experiments and build stack
experiments = itertools.product(problems, models, policies)
stack = OrderedDict()

for problem, model, policy in experiments:
    name = format(problem[0](**problem[1]))
    if name not in stack:
        stack[name] = OrderedDict()

    # name the particular run, in this example the relevant quantities are
    # the policy and the model
    key = str(policy[0])
    key += '-{0}'.format(str(model)) if model is not None else ''

    # the following kwargs must correspond to those required by the particular
    # experiment to be run
    stack[name][key] = dict(
        problem=problem,
        model=model,
        policy=policy,
        niter=niter,
    )

# run stack of tasks
results = run_stack(run_instance, stack, nreps)
