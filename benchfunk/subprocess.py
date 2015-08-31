import numpy as np
import re
import subprocess


__all__ = ['Subprocess']


class Subprocess(object):
    """
    Class for black-boxes that should be run from the shell. Simply
    pass the shell command with variables replaced with `{}` with
    python string formatting specs inside, then call the object
    with inputs to replace the `{}` in the same order as in the
    provided string.
    """
    def __init__(self, command):
        self.command = command

    def __call__(self, x):
        out = subprocess.check_output(self.command.format(*x),
                                      shell=True)
        out = out.splitlines()[-1]                      # keep last line
        out = re.compile(r'\x1b[^m]*m').sub('', out)    # strip color codes
        out = out.split('=')[-1]                        # strip left hand side

        return np.float(out)

    def get(self, X):
        raise NotImplementedError

    def get_f(self, X):
        raise NotImplementedError

