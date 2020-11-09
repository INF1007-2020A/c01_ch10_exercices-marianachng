#!/usr/bin/env python
# -*- coding: utf-8 -*-


# TODO: Importez vos modules ici
import numpy as np
from cmath import polar

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

def create_graph():
    pass

def evaluer_integral():
    pass

if __name__ == '__main__':
    # TODO: Appelez vos fonctions ici

    pass
