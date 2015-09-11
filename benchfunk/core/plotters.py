import numpy as np
from jug import TaskGenerator
import ezplot


__all__ = ['plot_stack']


@TaskGenerator
def plot_stack(stack_results, problems=None, policies=None, name=''):
    problems = problems if problems is not None else stack_results.key()
    nfigs = len(problems)

    fig = ezplot.figure(figsize=(5*nfigs, 4))

    for i, expt in enumerate(problems):
        results = stack_results[expt]
        policies = policies if policies is not None else results.keys()

        ax = fig.add_subplot(1, nfigs, i+1)

        for policy in policies:
            xbest, ybest = zip(*results[policy])
            iters = np.arange(np.shape(ybest)[1])

            mu = np.mean(ybest, axis=0)
            std = np.std(ybest, axis=0) / np.sqrt(len(ybest))
            ax.plot_banded(iters, mu, std, label=policy)

        ax.set_title(expt, fontsize=16)

    ax.legend(loc=0, fontsize=16)
    ezplot.plt.savefig(name)

    return fig

