import matplotlib.pyplot as plt
import numpy as np

plt.rc('lines', linewidth=2)

import pybo
import pybo.policies as pp

from jug import TaskGenerator
from jug.compound import CompoundTaskGenerator

from classics import *


@TaskGenerator
def run_instance(problem, policy, niter, seed):
    func = problem[0](rng=seed, **problem[1])           # instantiate problem
    bounds = func.bounds
    policy, policy_kwargs = policy                      # unpack policy

    model = pybo.init_model(func, bounds)               # initialize model
    ninit = model.ndata


    R = [pybo.recommenders.best_latent(model, bounds)]  # get recommendation

    for _ in xrange(ninit, niter):
        index = policy(model, bounds, **policy_kwargs)
        xnext, _ = pybo.solvers.solve_lbfgs(index, bounds)

        model.add_data(xnext, func(xnext))
        R.append(pybo.recommenders.best_latent(model, bounds))

    data = np.zeros(niter - ninit + 1,
                    [('R', np.float, (len(bounds),)),
                     ('F', np.float)])

    data['R'] = R
    data['F'] = func.get_f(data['R'])

    return data


@CompoundTaskGenerator
def run_experiment(problem, policies, niter, nreps, name=''):
    # data = {
    #     key: [solve_problem(problem, policy, niter, seed)
    #           for seed in xrange(nreps)]
    #     for key, policy in policies.items()
    # }
    data = dict()
    for key, policy in policies.items():
        data[key] = [run_instance(problem, policy, niter, seed)
                     for seed in xrange(nreps)]
        for t in data[key]:
            t.name = '.'.join([name, key])
    return data


@CompoundTaskGenerator
def run_stack(problems, policies, niter, nreps, name=''):
    # data = {
    #     key: run_experiment(problem, policies, niter, nreps)
    #     for key, problem in problems.items()
    # }
    data = dict()
    for key, problem in problems.items():
        namekey = '.'.join([name, key])
        data[key] = run_experiment(problem, policies, niter, nreps, namekey)
        data[key].name = namekey
    return data

@TaskGenerator
def plot_stack(stack_results, name=''):
    nfigs = len(stack_results.keys())
    fig, axs = plt.subplots(1, nfigs, sharey=True)

    for ax, (expt, results) in zip(axs, stack_results.items()):
        for k, v in results.items():
            # TODO: Fix this mess. Due to having a list of structured arrays.
            nreps = len(v)
            mean = v[0]['F']
            for i in xrange(1, nreps):
                mean += v[i]['F']

            ax.plot(mean, label=k)

        ax.set_title(expt, fontsize=16)

    axs[0].legend(loc=0, fontsize=16)
    plt.savefig(name, bbox_inches='tight')

    return fig


###############################################################################
# Run the experiments
###############################################################################

if True:
    # parameters
    name = __name__
    niter = 30
    nreps = 10

    problems = {
        'Gramacy(0.01)': (Gramacy, dict(sn2=0.01)),
        'Gramacy(0.05)': (Gramacy, dict(sn2=0.05)),
    }

    policies = {
        'EI(0.0)': (pp.EI, dict(xi=0.0)),
        'PI(0.1)': (pp.PI, dict(xi=0.1)),
        'TS(100)': (pp.Thompson, dict(n=100)),
    }

    results = run_stack(problems, policies, niter, nreps, name)
    results.name = name

    fig = plot_stack(results, 'foo.pdf')
    fig.name = '.'.join([name, 'plot'])
