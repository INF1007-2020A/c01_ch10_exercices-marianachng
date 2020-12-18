#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import math
import matplotlib.pyplot as plt
import numpy as np

from cmath import polar
from matplotlib.patches import Polygon
from scipy.integrate import quad
from typing import Callable


# TODO: Définissez vos fonctions ici (il en manque quelques unes)
def linear_values() -> np.ndarray:
    return np.linspace(-1.3, 2.5, 64)


def coordinate_conversion(cartesian_coordinates: np.ndarray) -> np.ndarray:
    a = np.zeros([len(cartesian_coordinates), 2])

    for i in range(len(cartesian_coordinates)):
        rho = np.sqrt(cartesian_coordinates[i][0] ** 2 + cartesian_coordinates[i][1] ** 2)
        phi = np.arctan2(cartesian_coordinates[i][1], cartesian_coordinates[i][0])
        polar_coordinate = (rho, phi)
        a[i] = polar_coordinate
        #a[i] = polar(cartesian_coordinates[i])

    return a

def coordinate_conversion_cmath(cartesian_coordinates: np.ndarray) -> np.ndarray:
    return np.array([polar(coord) for coord in cartesian_coordinates])

def find_closest_index(values: np.ndarray, number: float) -> int:
    return sorted([(i, values[i]) for i in range(values.size)], key= lambda element : abs(element[1] - number))[0][0]

def find_closest_indexx(values: np.ndarray, number: float) -> int:
    return np.abs(values - number).argmin()

def exercise_sin() -> None:
    graph_sinusoid_sample(*samples_from_function(sinusoid, -1,1,250))

def graph_sinusoid_sample(x: np.ndarray, y: np.ndarray):
    plt.plot(x, y, 'o', markersize = 2.5)
    plt.legend(['data'], loc = 'best')
    plt.show()

def samples_from_function(func: Callable, start: float, end: float, nb_samples: int) -> tuple:
    x = np.linspace(start, end, num = nb_samples, endpoint = True)
    y = np.array([func(x_i) for x_i in x])
    _ = sinusoid_np(x)
    return x, y

def sinusoid(x: float) -> float:
    return x**2 * math.sin(1/x**2) + x

def sinusoid_np(x: np.darray) -> np.darray:
    return x**2 * np.sin(1/x**2) + x

def create_graph():
    pass
def definite_intergral() -> tuple:
    return quad(integrand, -np.inf, np.inf)

def integrand(x: np.ndarray) -> np.ndarray:
    return np.exp(-x ** 2)

def integral(a,b):
    def f(x):
        return math.e ** (-1 * x ** 2)
    Ih, err = quad(f, a, b)

    return Ih, err
    
def evaluer_integral():
    pass

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici

    print(linear_values())
    exercise_sin()
