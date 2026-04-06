import numpy as np
from random import randint


def to_grayscale_pixel(pixel):
    return np.uint8(round(np.mean(pixel)))


def to_grayscale_image(image):
    rows, cols, _other = np.shape(image)
    gray_image = np.zeros((rows, cols), dtype=np.uint8)
    for i in range(rows):
        for j in range(cols):
            gray_image[i, j] = to_grayscale_pixel(image[i, j])
    return gray_image


def flatten_2d(image):
    values = []
    for row in image:
        for value in row:
            values.append(value)
    return values


def inject_localized_bit_errors(encoded_image, error_locations, max_bit_index):
    # The structure can be ragged (list of blocks), so avoid global np.copy.
    altered_image = [np.array(block, copy=True) for block in encoded_image]
    for index in error_locations:
        bit_index = randint(0, max_bit_index)
        if altered_image[index][bit_index]:
            altered_image[index][bit_index] = 0
        else:
            altered_image[index][bit_index] = 1
    return altered_image

