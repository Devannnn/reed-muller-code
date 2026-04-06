import numpy as np
from random import randint


def gris(pixel):
    return np.uint8(round(np.mean(pixel)))


def conversion(image):
    ligne, colonne, _autre = np.shape(image)
    image_gris = np.zeros((ligne, colonne), dtype=np.uint8)
    for i in range(ligne):
        for j in range(colonne):
            image_gris[i, j] = gris(image[i, j])
    return image_gris


def aplatir(image):
    values = []
    for row in image:
        for value in row:
            values.append(value)
    return values


def creation_erreurs_localises(encoded_image, localisation_erreurs, max_bit_index):
    image_alteree = [np.array(bloc, copy=True) for bloc in encoded_image]
    for index in localisation_erreurs:
        bit_index = randint(0, max_bit_index)
        if image_alteree[index][bit_index]:
            image_alteree[index][bit_index] = 0
        else:
            image_alteree[index][bit_index] = 1
    return image_alteree
