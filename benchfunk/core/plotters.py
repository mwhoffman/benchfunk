import matplotlib.pyplot as plt
from jug import TaskGenerator

plt.rc('lines', linewidth=2)


__all__ = ['plot_stack']


@TaskGenerator
def plot_stack(stack_results, problems=None, policies=None, name=''):
    problems = problems if problems is not None else stack_results.key()
    nfigs = len(problems)

    fig, axs = plt.subplots(1, nfigs, sharey=True)

    for ax, expt in zip(axs, problems):
        results = stack_results[expt]
        policies = policies if policies is not None else results.keys()

        for policy in policies:
            v = results[policy]
            # TODO: Fix this mess. Due to having a list of structured arrays.
            nreps = len(v)
            mean = v[0][1]
            for i in xrange(1, nreps):
                mean += v[i][1]
            mean /= nreps

            ax.plot(mean, label=policy)

        ax.set_title(expt, fontsize=16)

    axs[0].legend(loc=0, fontsize=16)
    plt.savefig(name)

    return fig

