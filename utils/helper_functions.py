import csv

import numpy as np


def print_2d_array(array):
    """
    Printing of a 2D array

    array: 2D array
    """

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            print(array[i][j], end=" ")
        print()


def write_1d_array_in_file(array, filename=""):
    """
    Writes 1D array in file.

    array: 1D array
    filename: string
    """

    f = open(filename, "w")

    for i in range(len(array)):
        f.write('%f\n' % array[i])

    f.close()


def write_2d_array_in_file(array, filename=""):
    """
    Writes 2D array in file.

    array: 2D array
    filename: string
    """

    f = open(filename, "w")

    for i in range(array.shape[0]):
        for j in range(array.shape[1]):
            f.write('%f ' % array[i][j])
        f.write('\n')

    f.close()


def read_2d_array_from_file(filename="", n=79, m=79):
    """
    Read 2D array of size n x m from file.

    filename: string
    n: integer, rows
    m: integer, columns
    """

    f = open(filename, "r")
    matrix = f.read()
    rows = [item.split() for item in matrix.split('\n')]

    matr = np.zeros((n, m))
    for i, row in enumerate(rows):
        for j, elem in enumerate(row):
            matr[i][j] = round(float(elem), 3)

    f.close()

    return matr


def find_min_max_from_2d_array(matrix):
    """
    Finds the minimum, maximum values and the index of maximum value.

    @param matrix: 2D array
    @return: minimum, maximum values and the index of maximum value
    """

    min_value = np.min(matrix)
    max_value = np.max(matrix)

    max_value_index = np.unravel_index(np.argmax(matrix, axis=None), matrix.shape)

    return min_value, max_value, max_value_index


def write_in_csv(array, filename=""):
    header_labeled_data = ['Feature number', 'Distance']
    rows = [header_labeled_data]

    for tup in array:
        rows.append([tup[0]] + [tup[1]])

    with open(filename, 'w', newline='') as file:
        writer = csv.writer(file, delimiter=',')
        writer.writerows(rows)


def create_dirs():
    import os
    for i in range(0, 95):
        os.mkdir('figs/validation/sim' + str(i))

