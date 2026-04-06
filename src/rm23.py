## MODULES
import numpy as np
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


## DECODING


# Build characteristic vectors for a degree-2 monomial on row i.
def degree2_characteristic_vectors(r, m, i):
    rm_matrix = build_rm_generator_matrix(r, m)
    vector = rm_matrix[len(rm_matrix) - i]
    return [vector, bitwise_complement(vector)]


def index_pairs(m):
    pairs = []
    for i in range(0,2*m-2,m-1):
        for j in range(1,2*m-2):
            if j!=i+m-1 and i!=j :
                pairs.append((i,j))
    return pairs


# Build characteristic vectors for a degree-1 monomial on row i.
def degree1_characteristic_vectors(r, m, i):
    vectors = []
    hyperplane_vectors = build_hyperplanes(m)
    characteristic = []
    pair_indices = index_pairs(m)

    # Keep vectors v_j where i != j.
    for j in range(1, len(hyperplane_vectors)):
        if j!=i:
            vectors.append(hyperplane_vectors[j])

    # Append their bitwise complements.
    for k in range(len(vectors)):
        vectors.append(bitwise_complement(vectors[k]))

    # Multiply selected pairs returned by index_pairs().
    vectors = np.array(vectors)
    for X in pair_indices:
        a,b = X
        characteristic.append(vectors[a] * vectors[b])
    return characteristic


def index_triplets():
    triplets = []
    for i in range(2):
        for j in range(2,4):
            for h in range(4,6):
                triplets.append((i,j,h))
    return triplets


# Build characteristic vectors for the degree-0 monomial (first row).
def degree0_characteristic_vectors(r, m):
    vectors = build_hyperplanes(m)[1:]
    all_vectors = []
    characteristic = []
    triplet_indices = index_triplets()

    for i in range(m):
        all_vectors.append(vectors[i])
        all_vectors.append(bitwise_complement(vectors[i]))

    # Multiply selected triplets returned by index_triplets().
    all_vectors = np.array(all_vectors)
    for X in triplet_indices:
        a,b,c = X
        characteristic.append(all_vectors[a] * all_vectors[b] * all_vectors[c])
    return characteristic


# Build the complete characteristic-vector table once.
def build_characteristic_vectors(r, m):
    characteristic_vectors = [degree0_characteristic_vectors(2, 3)]
    for i in range(1,m):
        characteristic_vectors.append(degree1_characteristic_vectors(2, 3, i))
    for j in range(m,7):
        characteristic_vectors.append(degree2_characteristic_vectors(2, 3, j))
    return characteristic_vectors

characteristic_vectors = tuple(build_characteristic_vectors(2, 3))

# Process row i with majority vote.
def majority_vote(r, m, i, message):
    characteristic_list = characteristic_vectors[i]
    extraction = 0
    for X in characteristic_list:
        extraction += dot_product_mod2(message, X)
    if extraction < len(characteristic_list)//2:
        return 0
    else:
        return 1

# Decode one vector with majority-vote decoding (RM(2,3)).
def decode_vector(r, m, message):
    rm_matrix = build_rm_generator_matrix(r, m)
    message_decode = np.array([None for i in range(len(rm_matrix))])

    # Calcul des monômes de degré 2
    for i in range(len(rm_matrix)-1,m,-1):
        message_decode[i] = majority_vote(r, m, i, message)

    # Les monômes de degre 2 ont été traités : on modifie le message
    S1 = vector_matrix_product_mod2(message_decode[m+1:], rm_matrix[m+1:])
    E1 = add_vectors_mod2(S1, message)

    # On traite ensuite les monômes de degré 1
    for j in range(m,0,-1):
        message_decode[j] = majority_vote(r, m, j, E1)

    # Les monômes de degré 1 ont été traités : on modifie le message
    S2 = vector_matrix_product_mod2(message_decode[1:4], rm_matrix[1:4])
    E2 = add_vectors_mod2(S2, E1)

    # Calcul des monômes de degre 0
    message_decode[0] = majority_vote(r, m, 0, E2)
    return message_decode


## MESSAGE ENCODING

# Convert an image (pixel matrix) into one flat list of bits.
def pixels_to_bitstream(image):
    # Split pixels and convert each one to 8 bits.
    L=[]
    for X in image:
        for Y in  X:
            L.append(decimal_to_binary(8,Y))
    # We handle 7-bit message chunks, so flatten all per-pixel bit lists.
    H = []
    for X in L:
        for Y in  X:
            H.append(Y)
    return H


# Split an image bitstream into 7-bit blocks for RM(2,3).
def split_into_blocks(image):
    L = []
    copie = pixels_to_bitstream(image)
    taille = len(copie)
    for k in range(0,taille,7):
            L.append(copie[k:k+7])
    return L


# Encode an image by processing each 7-bit block.
def encode_image(image):
    liste_pixels = split_into_blocks(image)
    RM = build_rm_generator_matrix(2,3)
    for i in range(len(liste_pixels)-1):
        liste_pixels[i] = vector_matrix_product_mod2(liste_pixels[i],RM)
    return liste_pixels


# Inverse operation of split_into_blocks: rebuild pixel matrix.
def reassemble_image(image,hauteur,largeur):
    copie,L = flatten_2d(image), []

    # Convert each 8-bit chunk back to an integer pixel.
    for k in range(0,len(copie),8):
        L.append(binary_to_decimal(copie[k:k+8]))

    matrice = np.zeros((hauteur,largeur), dtype=np.uint8 )
    indice_ligne = 0

    # Rebuild the original image matrix.
    for x in range(0,len(L)):
        matrice[indice_ligne][x%largeur] = L[x]
        if x!=0 and not(x%largeur):
            indice_ligne +=1
    return matrice


# Inverse operation of encode_image on 8-bit codewords.
def decode_image(image,hauteur,largeur):
    image_decode  = []

    # Decode each vector with RM(2,3).
    for X in image:
        if len(X)==8:
            image_decode.append(decode_vector(2,3,X))
        else:
            image_decode.append(X)

    # Rebuild the image.
    image_decode = reassemble_image(image_decode,hauteur,largeur)
    return np.array(image_decode)



## ERROR INJECTION

# Randomly inject pixel errors and keep their locations.
def inject_errors(image,nb_erreurs):
    lignes,colonnes = np.shape(image)
    localisation_erreurs = []
    image_alteree = np.copy(image)

    for i in range(nb_erreurs):
        e_ligne,e_colonne,couleur = randint(0, lignes-1),randint(0, colonnes-1),randint(0, 255)
        localisation_erreurs.append(    e_ligne*colonnes + (e_colonne-1)  )
        image_alteree[e_ligne][e_colonne] = couleur

    return image_alteree,localisation_erreurs

# Inject bit errors on encoded blocks from known locations.
def inject_localized_errors(image,localisation_erreurs):
    return _inject_localized_bit_errors(image, localisation_erreurs, 7)


## ENCODING/DECODING TESTS

RM = build_rm_generator_matrix(2,3)

v1=np.array([1, 1, 1, 1, 0, 0, 0 ])
v2=np.array([1, 0, 1, 0, 1, 0, 1 ])

message = np.array([0,1,1,1,0,1,0])
message_encode = vector_matrix_product_mod2(message,RM)
e1 = majority_vote(2,3,1,message_encode)
d1 = decode_vector(2,3,message_encode)


message2 = np.array([0, 0, 1, 0, 0, 0, 1])
message_encode2 = vector_matrix_product_mod2(message2,RM)
e2 = majority_vote(2,3,2,message_encode2)
d2 = decode_vector(2,3,message_encode2)

test =  np.array([[100,105,200],[45,204,0],[94,0,204]])

a = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 0, 0, 0]])

b = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 0, 0, 0]])


res1 = split_into_blocks(a)
res2 = encode_image(res1)
res3 = []

for X in res2:
    if len(X)==8:
        res3.append(decode_vector(2,3,X))
    else:
        res3.append(X)


## ERROR-GENERATION TEST


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
surfeur          = plt.imread(DATA_DIR / "surfer.jpg")
surfeur_gris     = to_grayscale_image(surfeur)


def compute_demo():
    surfeur_encode = encode_image(surfeur_gris)
    surfeur_decode0 = decode_image(surfeur_encode,427,640)


    surfeur_alt1,l1  = inject_errors(surfeur_gris,100)
    surfeur_alt2,l2  = inject_errors(surfeur_gris,1000)

    surfeur_ealt1    = inject_localized_errors(surfeur_encode,l1)
    surfeur_ealt2    = inject_localized_errors(surfeur_encode,l2)


    surfeur_decode1 = decode_image(surfeur_ealt1,427,640)
    surfeur_decode2 = decode_image(surfeur_ealt2,427,640)


    return [ surfeur_gris    , surfeur_alt1    , surfeur_alt2    ,
            surfeur_decode0 , surfeur_decode1 , surfeur_decode2 ]


def display_images(images):
    lignes , colonnes = 2, 3
    axes=[]
    fig=plt.figure()

    for a in range(lignes*colonnes):
        axes.append( fig.add_subplot(lignes, colonnes, a+1) )
        plt.axis('off')
        plt.title(str(a))
        plt.imshow(images[a], cmap='gray')


    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    display_images(compute_demo())







