## MODULES
import numpy as np
from copy import deepcopy
from random import randint
import matplotlib.pyplot as plt
from pathlib import Path
from core import (
    binary_to_decimal,
    decimal_to_binary,
    build_hyperplanes,
    build_rm_generator_matrix,
    dot_product_mod2,
    vector_matrix_product_mod2,
    add_vectors_mod2,
    bitwise_complement,
)
from image import (
    to_grayscale_image,
    flatten_2d,
    inject_localized_bit_errors as _inject_localized_bit_errors,
)


RM = build_rm_generator_matrix(2,4)


## DECODING


# Build characteristic vectors for a degree-2 monomial on row i.
def degree2_characteristic_vectors(r,m,i):
    indices = []
    vecteur = []
    cara = []

    # Determine degree-1 monomials not used to build RM[i].
    if i>=8 and i<=10:
        indices.append(1)
        indices.append(12-i)
    elif i>=6 and i<=7:
        indices.append(2)
        indices.append(10-i)
    else:
        indices.append(3)
        indices.append(4)

    # Append complements.
    for k in range(len(indices)):
        cara.append(RM[indices[k]])
        cara.append(bitwise_complement(RM[indices[k]]))

    couples = [(0,2),(0,3),(1,2),(1,3)]
    for X in couples :
        a,b = X
        vecteur.append(cara[a]*cara[b])
    return vecteur


def degree1_triplets():
    P=[]
    for a in range(0,2):
        for b in range(2,4):
            for c in range(4,6):
                P.append((a,b,c))
    return P


# Build characteristic vectors for a degree-1 monomial on row i.
def degree1_characteristic_vectors(r,m,i):
    L=[]
    v = build_hyperplanes(m)
    caracteristique = []

    # Keep vectors v_j where i != j.
    for j in range(1,len(v)):
        if j!=i:
            L.append(v[j])
            L.append(bitwise_complement(v[j]))

    L = np.array(L)
    couples = degree1_triplets()
    for X in couples :
        a,b,c = X
        caracteristique.append(L[a]*L[b]*L[c])
    return caracteristique


def degree0_triplets():
    L = []
    for i in range(2):
        for j in range(2,4):
            for h in range(4,6):
                L.append((i,j,h))
    return L


# Build characteristic vectors for degree-0 monomial (first row).
def degree0_characteristic_vectors(r,m):
    v = np.eye(2**m)
    caracteristique = []
    for X in range(2**m):
        caracteristique.append(v[0])
    return caracteristique


# Build the complete characteristic-vector table once.
def build_characteristic_vectors(r,m):
    vecteur_caracteristique = [degree0_characteristic_vectors(r,m)]
    for i in range(1,m+1):
        vecteur_caracteristique.append(degree1_characteristic_vectors(r,m,i))
    for j in range(m+1,11):
        vecteur_caracteristique.append(degree2_characteristic_vectors(r,m,j))
    return vecteur_caracteristique

vecteur_caracteristique = tuple(build_characteristic_vectors(2,4))


# Process row i with majority vote.
def majority_vote(r,m,i,message):
    liste_v_cara = vecteur_caracteristique[i]
    extraction = 0
    for X in liste_v_cara:
        extraction += dot_product_mod2(message,X)
    if extraction < len(liste_v_cara)//2:
        return 0
    else:
        return 1

RM = build_rm_generator_matrix(2,4)
# Decode one vector with majority-vote decoding (RM(2,4)).
def decode_vector(r,m,message):
    message_decode = np.array([None for i in range(len(RM))])

    # Calcul des monômes de degré 2
    for i in range(len(RM)-1,m,-1):
        message_decode[i] = majority_vote(r,m,i,message)

    # Les monômes de degre 2 ont été traités : on modifie le message
    S1 = vector_matrix_product_mod2(message_decode[m+1:],RM[m+1:])
    E1 = add_vectors_mod2(S1,message)

    # On traite ensuite les monômes de degré 1
    for j in range(m,0,-1):
        message_decode[j] = majority_vote(r,m,j,E1)

    # Les monômes de degré 1 ont été traités : on modifie le message
    S2 = vector_matrix_product_mod2(message_decode[1:m+1],RM[1:m+1])
    E2 = add_vectors_mod2(S2,E1)

    # Calcul des monômes de degre 0
    message_decode[0] = majority_vote(r,m,0,E2)
    return message_decode



## MESSAGE ENCODING

# Convert image pixels into 11-bit vectors.
def split_into_blocks(image):
    # Split pixels and convert each one to 11 bits.
    L=[]
    for X in image:
        for Y in  X:
            L.append(decimal_to_binary(11,Y))
    return L


# Inverse operation of split_into_blocks.
def reassemble_image(image,hauteur,largeur):
    IMAGE_COPY = flatten_2d(image)
    INT_LIST = []

    # Convert each 11-bit vector back to its integer value.
    for k in range(0,len(IMAGE_COPY),11):
        INT_LIST.append(binary_to_decimal(IMAGE_COPY[k:k+11]))

    matrice = np.zeros((hauteur,largeur), dtype=np.uint8 )
    indice_ligne = 0

    # Rebuild the original image matrix.
    for x in range(0,len(INT_LIST)):
        if x!=0 and not(x%largeur):
            indice_ligne +=1
        matrice[indice_ligne][x % largeur] = INT_LIST[x]
    return matrice


# Encode an image by processing each 11-bit vector.
def encode_image(image):
    liste_pixels = split_into_blocks(image)
    RM = build_rm_generator_matrix(2,4)
    for i in range(len(liste_pixels)):
        liste_pixels[i] = vector_matrix_product_mod2(liste_pixels[i],RM)
    return liste_pixels


# Inverse operation of encode_image for RM(2,4).
def decode_image(IMAGE, HEIGHT, WIDTH):
    IMAGE_DECODED  = []

    # Decode each vector with RM(2,4).
    for VECTOR in IMAGE:
        if len(VECTOR) == 16:
            IMAGE_DECODED.append(decode_vector(2,4,VECTOR))
        else:
            IMAGE_DECODED.append(VECTOR)
    # Rebuild the image.
    IMAGE_DECODED = reassemble_image(IMAGE_DECODED, HEIGHT, WIDTH)
    return np.array(IMAGE_DECODED)



## ERROR INJECTION


# Randomly inject pixel errors and keep their locations.
def inject_errors(image, nb_erreurs):
    lignes,colonnes = np.shape(image)
    localisation_erreurs = []
    image_alteree = deepcopy(image)

    for i in range(nb_erreurs):
        e_ligne,e_colonne,couleur = randint(0, lignes-1),randint(0, colonnes-1),randint(0, 255)
        localisation_erreurs.append( e_ligne*colonnes + e_colonne )
        image_alteree[e_ligne][e_colonne] = couleur

    return image_alteree,localisation_erreurs


# Inject bit errors on encoded blocks from known locations.
def inject_localized_errors(image, localisation_erreurs):
    return _inject_localized_bit_errors(image, localisation_erreurs, 15)


# Inject at most one error per 16-bit block.
def inject_optimized_errors(image, erreurs):
    lignes,colonnes = np.shape(image)
    localisation_erreurs = []
    image_alteree = np.copy(image)
    nb_erreurs = min(lignes*colonnes,erreurs)

    for i in range(nb_erreurs):

        e_ligne,e_colonne,couleur = randint(0, lignes-1),randint(0, colonnes-1),randint(0, 255)

        while ((e_ligne*colonnes + e_colonne) in localisation_erreurs):
            e_ligne,e_colonne,couleur = randint(0, lignes-1),randint(0, colonnes-1),randint(0, 255)

        localisation_erreurs.append( e_ligne*colonnes + e_colonne )
        image_alteree[e_ligne][e_colonne] = couleur

    return image_alteree,localisation_erreurs




## TEST


def compute_joconde_demo():
    RESULTS       = []
    HEIGHT        = 417
    WIDTH         = 300
    DATA_DIR      = Path(__file__).resolve().parent.parent / "data"
    IMAGE_PATH    = DATA_DIR / "joconde.jpg"
    IMAGE         = plt.imread(IMAGE_PATH)
    IMAGE_GREY    = to_grayscale_image(IMAGE)
    IMAGE_ENCODED = encode_image(IMAGE_GREY)
    NUMBER_OF_ERRORS = [100, 1000, 10000, 25000, 50000]

    for NUMBER in NUMBER_OF_ERRORS:
        IMAGE_ALTERED, ERRORS_LOCATION  = inject_errors(IMAGE_GREY, NUMBER)
        IMAGE_ENCODED_AND_ALTERED = inject_localized_errors(IMAGE_ENCODED, ERRORS_LOCATION)
        IMAGE_DECODED = decode_image(IMAGE_ENCODED_AND_ALTERED, HEIGHT, WIDTH)
        RESULTS.append([IMAGE_ALTERED, IMAGE_DECODED])

    return RESULTS



def display_images(IMAGES):
    LINES = 2
    COLUMNS = len(IMAGES) // 2
    axes=[]
    fig=plt.figure()
    for a in range(LINES*COLUMNS):
        axes.append( fig.add_subplot(LINES, COLUMNS, a+1) )
        plt.axis('off')
        plt.title(str(a))
        plt.imshow(IMAGES[a], cmap='gray')
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    display_images(compute_joconde_demo())






