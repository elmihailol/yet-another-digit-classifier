import numpy


def add_noise(x, noise_factor=0.1):
    """
    Добавляет шум в матрицу
    :param x: оригинальная матрица (картинка)
    :param noise_factor: фактор шума
    :return: картинка (матрица) с шумом
    """
    x = x + numpy.random.randn(*x.shape) * noise_factor
    x = x.clip(0., 5.)
    return x
