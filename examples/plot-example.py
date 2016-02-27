import benchfunk
import matplotlib.pyplot as plt
import numpy as np
from example import results
import jug

jug.set_jugdir('example.jugdata')

fig, axs = plt.subplots(1, len(results),
                        figsize=(5*len(results), 4),
                        sharex=True)

for ax, (name, res) in zip(axs, results.items()):
    func, sn2 = name.split('(')
    sn2 = float(sn2[:-1])
    obj = getattr(benchfunk.functions, func)(sn2)

    for key, xbest in res.items():
        xbest = jug.value(xbest)
        xbest = np.array(xbest)
        ybest = np.array([obj.get_f(run) for run in xbest])
        ax.plot(ybest.mean(0), label=key)

    ax.set_title(name)

plt.show()
