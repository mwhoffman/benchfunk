import numpy as np


__all__ = ['Interactive']


class Interactive(object):
    def __init__(self, prompt='Enter value at design x = {}\ny = '):
        self.prompt = prompt

    def __call__(self, x):
        y = input(self.prompt.format(x))
        assert isinstance(y, (np.int, np.long, np.float)), \
            'output must be a number'
        return y

    def get(self, X):
        raise NotImplementedError

    def get_f(self, X):
        raise NotImplementedError
