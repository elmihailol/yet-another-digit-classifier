import numpy


def add_noise(x, noise_factor=0.1):
    x = x + numpy.random.randn(*x.shape) * noise_factor
    x = x.clip(0., 5.)
    return x
