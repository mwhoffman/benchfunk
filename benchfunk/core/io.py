"""
Input-output functions for dumping and loading results from launched
experiments.
"""

from collections import OrderedDict
import cPickle as pickle
import imp
import jug
import os.path

__all__ = ['dump', 'load']


def dump(jugfile):
    # get absolute path and import the Tasks from the associated source code
    jugpath, ext = os.path.splitext(os.path.abspath(jugfile))
    modulename = os.path.basename(jugpath)

    # set path to results and load execution script
    # NOTE: jugdir must be set *before* loading the source
    jug.set_jugdir(jugpath + '.jugdata')
    results = imp.load_source(modulename, jugpath + ext).results

    # initialize dictionary of completed results to be incrementally built below
    completed = OrderedDict()
    incomplete = False

    for name, experiment in results.items():
        completed[name] = OrderedDict()

        for key, task in experiment.items():
            completed[name][key] = []

            for run in task:
                try:
                    completed[name][key].append(jug.value(run))
                except:
                    incomplete = True
                    pass

            # get rid of empty keys
            if len(completed[name][key]) == 0:
                completed[name].pop(key)

        if not completed[name]:
            completed.pop(name)

    # if dictionary is not empty, dump to pickle file
    if completed:
        if incomplete:
            jugpath += '.tmp'
        with open(jugpath + '.pkl', 'w') as fp:
            pickle.dump(completed, fp)

    return completed


def load(pklfile):
    with open(pklfile, 'r') as fp:
        results = pickle.load(fp)

    return results