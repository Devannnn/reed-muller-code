import numpy as np


def base10(bits):
    value = 0
    for i in range(len(bits)):
        value += bits[i] * 2 ** (len(bits) - i - 1)
    return value


def conv_base_2(n, b):
    bits = []
    while b > 0:
        bits.append(b % 2)
        b = b // 2
    for _ in range(n - len(bits)):
        bits.append(0)
    bits.reverse()
    return bits


def F2_m(m):
    values = []
    for i in range(2 ** m):
        values.append(conv_base_2(m, i))
    return values


def hyperplans(m):
    f2 = F2_m(m)
    plans = [[1 for _ in range(2 ** m)]]
    for j in range(m):
        current = []
        for i in range(2 ** m):
            if f2[i][j]:
                current.append(0)
            else:
                current.append(1)
        plans.append(current)
    return plans


def produit_exterieur(vectors, i, j):
    values = []
    for k in range(len(vectors[i])):
        values.append(vectors[i][k] * vectors[j][k])
    return values


def matrice_generatrice_RM(r, m):
    vectors = hyperplans(m)
    if r == 1:
        return np.array(vectors)
    if r == 2:
        matrix = vectors.copy()
        for i in range(1, m):
            for j in range(1, m + 1 - i):
                matrix.append(produit_exterieur(vectors, i, i + j))
        return np.array(matrix)
    return "erreur"


def prod_scal_mod2(v1, v2):
    n = len(v1)
    scalar = 0
    for i in range(n):
        scalar = scalar + v1[i] * v2[i]
    return scalar % 2


def produit_vecteur_matrice(vector, matrix):
    _m, p = matrix.shape
    result = np.array([0 for _ in range(p)])
    for j in range(p):
        result[j] = prod_scal_mod2(vector, matrix[:, j])
    return result


def addition_mod2(v1, v2):
    size = len(v1)
    summed = np.array([0 for _ in range(size)])
    for i in range(size):
        summed[i] = (v1[i] + v2[i]) % 2
    return summed


def complementaire(vector):
    comp = vector.copy()
    for i in range(len(comp)):
        if comp[i]:
            comp[i] = 0
        else:
            comp[i] = 1
    return comp
