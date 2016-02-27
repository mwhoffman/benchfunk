# benchfunk

A python package to facilitate benchmarking algorithms.

This package provides _(i)_ a set of test functions that can be used to
benchmark global optimization and Bayesian optimization methods;
and _(ii)_ a framework to facilitate running repeated experiments
(powered by `jug`).

## example

As an example, from the `examples/` directory, run:
```
jug execute example.py &
jug status example.py
```
where the second line allows you to track the progress.
Once completed, results can be manipulated and plotted; for an example,
run:
```
python plot-example.py
```
to visualize a sample output.
