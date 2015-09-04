from jug import TaskGenerator
from jug.compound import CompoundTaskGenerator
import pybo


__all__ = ['run_instance', 'run_experiment', 'run_stack']


@TaskGenerator
def run_instance(problem, policy, niter, seed):
    func = problem[0](rng=seed, **problem[1])           # instantiate problem
    bounds = func.bounds

    model = pybo.init_model(func, bounds)               # initialize model
    data, model = pybo.solve_bayesopt(func,
                               bounds,
                               model,
                               niter,
                               policy=policy,
                               recommender='incumbent')

    data['F'] = func.get_f(data['xbest'])

    return data


@CompoundTaskGenerator
def run_experiment(problem, policies, niter, nreps, script=None, name=''):
    data = dict()
    script = run_instance if script is None else script

    for key, policy in policies.items():
        namekey = '.'.join([name, key])

        data[key] = [script(problem, policy, niter, seed)
                     for seed in xrange(nreps)]

        for t in data[key]:
            t.name = namekey

    return data


@CompoundTaskGenerator
def run_stack(problems, policies, niter, nreps, script=None, name=''):
    data = dict()
    script = run_instance if script is None else script

    for key, problem in problems.items():
        namekey = '.'.join([name, key])

        data[key] = run_experiment(
            problem,
            policies,
            niter,
            nreps,
            script=script,
            name=namekey)

        data[key].name = namekey

    return data
