import os
from .vtk_functions import read_vtk


def get_example_knee(example_filename='change_cart_thick.vtk'):
    path = os.path.dirname(__file__)
    example_knee_mesh = read_vtk(os.path.join(path, 'data', example_filename))
    return example_knee_mesh
