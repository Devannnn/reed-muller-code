## MODULES
import numpy as np
from copy import deepcopy
from random import randint
import matplotlib.pyplot as plt
from core import (
    base10,
    conv_base_2,
    hyperplans,
    matrice_generatrice_RM,
    prod_scal_mod2,
    produit_vecteur_matrice,
    addition_mod2,
    complementaire,
)
from image import conversion, aplatir, creation_erreurs_localises as _creation_erreurs_localises


RM = matrice_generatrice_RM(2,4)


## DECODAGE


# Fonction qui détermine les vecteurs caractéristiques d'un monôme de degré 2 dont la ligne est la ligne i.
def vecteur_caracteristique_degre_2(r,m,i):
    indices = []
    vecteur = []
    cara = []

    #on détermine les monômes de degré 1 qui n'ont pas été utilisés pour créer RM[i]
    if i>=8 and i<=10:
        indices.append(1)
        indices.append(12-i)
    elif i>=6 and i<=7:
        indices.append(2)
        indices.append(10-i)
    else:
        indices.append(3)
        indices.append(4)

    #ajout du complémentaire
    for k in range(len(indices)):
        cara.append(RM[indices[k]])
        cara.append(complementaire(RM[indices[k]]))

    couples = [(0,2),(0,3),(1,2),(1,3)]
    for X in couples :
        a,b = X
        vecteur.append(cara[a]*cara[b])
    return vecteur


def couples_degre_1():
    P=[]
    for a in range(0,2):
        for b in range(2,4):
            for c in range(4,6):
                P.append((a,b,c))
    return P


# Fonction qui détermine les vecteurs caractéristiques d'un monôme de degré 1 dont la ligne est la ligne i.
def vecteur_caracteristique_degre_1(r,m,i):
    L=[]
    v = hyperplans(m)
    caracteristique = []

    #sépare les vecteurs v_j avec i!=j
    for j in range(1,len(v)):
        if j!=i:
            L.append(v[j])
            L.append(complementaire(v[j]))

    L = np.array(L)
    couples = couples_degre_1()
    for X in couples :
        a,b,c = X
        caracteristique.append(L[a]*L[b]*L[c])
    return caracteristique


def couples_degre_0():
    L = []
    for i in range(2):
        for j in range(2,4):
            for h in range(4,6):
                L.append((i,j,h))
    return L


# Fonction qui détermine les vecteurs caractéristiques du monôme de degré 0 correspondant à la première ligne de la matrice.
def vecteur_caracteristique_degre_0(r,m):
    v = np.eye(2**m)
    caracteristique = []
    for X in range(2**m):
        caracteristique.append(v[0])
    return caracteristique


# Fonction qui créée la liste complète des vecteurs caractéristiques de chaque monôme de la matrice, elle est appelée une et une seule fois d'où l'utilisation d'un tuple pour assurer l'intégrité des données pendant l'exécution du programme.
def creation_vecteur_caracteristique(r,m):
    vecteur_caracteristique = [vecteur_caracteristique_degre_0(r,m)]
    for i in range(1,m+1):
        vecteur_caracteristique.append(vecteur_caracteristique_degre_1(r,m,i))
    for j in range(m+1,11):
        vecteur_caracteristique.append(vecteur_caracteristique_degre_2(r,m,j))
    return vecteur_caracteristique

vecteur_caracteristique = tuple(creation_vecteur_caracteristique(2,4))


# Fonction qui traite la ligne i
def vote_majoritaire(r,m,i,message):
    liste_v_cara = vecteur_caracteristique[i]
    extraction = 0
    for X in liste_v_cara:
        extraction += prod_scal_mod2(message,X)
    if extraction < len(liste_v_cara)//2:
        return 0
    else:
        return 1

RM = matrice_generatrice_RM(2,4)
# Fonction qui décode, avec la méthode du vote majoritaire, un vecteur préalablement encodé par la matrice génératrice du code RM(r,m), ici il n'est efficace que pour le code RM(2,4).
def decodage_vecteur(r,m,message):
    message_decode = np.array([None for i in range(len(RM))])

    # Calcul des monômes de degré 2
    for i in range(len(RM)-1,m,-1):
        message_decode[i] = vote_majoritaire(r,m,i,message)

    # Les monômes de degre 2 ont été traités : on modifie le message
    S1 = produit_vecteur_matrice(message_decode[m+1:],RM[m+1:])
    E1 = addition_mod2(S1,message)

    # On traite ensuite les monômes de degré 1
    for j in range(m,0,-1):
        message_decode[j] = vote_majoritaire(r,m,j,E1)

    # Les monômes de degré 1 ont été traités : on modifie le message
    S2 = produit_vecteur_matrice(message_decode[1:m+1],RM[1:m+1])
    E2 = addition_mod2(S2,E1)

    # Calcul des monômes de degre 0
    message_decode[0] = vote_majoritaire(r,m,0,E2)
    return message_decode



## Encodage du message

# Fonction qui convertit une image, i.e , une matrice de pixels en une liste de liste de bits.
def decoupage(image):
    # on sépare les pixels et on en fait une liste
    L=[]
    for X in image:
        for Y in  X:
            # chaque pixel est codé sur 11 bits
            L.append(conv_base_2(11,Y))
    return L


# Fonction réciproque de decoupage, elle prend en entrée une liste de vecteurs qui correspondent aux listes de 11 bits précédemment citées, qui ont été décodées.
def recollage(image,hauteur,largeur):
    IMAGE_COPY = aplatir(image)
    INT_LIST = []

    # reconversion de chaque liste de 11 bits en l'entier correspondant
    for k in range(0,len(IMAGE_COPY),11):
        INT_LIST.append(base10(IMAGE_COPY[k:k+11]))

    matrice = np.zeros((hauteur,largeur), dtype=np.uint8 )
    indice_ligne = 0

    # restauration de la matrice de départ qui représente l'image.
    for x in range(0,len(INT_LIST)):
        if x!=0 and not(x%largeur):
            indice_ligne +=1
        matrice[indice_ligne][x % largeur] = INT_LIST[x]
    return matrice


# Fonction qui encode une image en traitant chaque liste de 11 bits.
def encodage(image):
    liste_pixels = decoupage(image)
    RM = matrice_generatrice_RM(2,4)
    for i in range(len(liste_pixels)):
        liste_pixels[i] = produit_vecteur_matrice(liste_pixels[i],RM)
    return liste_pixels


# Fonction réciproque de encodage, elle prend en entrée une liste de vecteurs de 11 bits.
def decodage(IMAGE, HEIGHT, WIDTH):
    IMAGE_DECODED  = []

    # chaque vecteur est décodé grâce au code RM(2,4)
    for VECTOR in IMAGE:
        if len(VECTOR) == 16:
            IMAGE_DECODED.append(decodage_vecteur(2,4,VECTOR))
        else:
            IMAGE_DECODED.append(VECTOR)
    # l'image est reformée
    IMAGE_DECODED = recollage(IMAGE_DECODED, HEIGHT, WIDTH)
    return np.array(IMAGE_DECODED)



## Génération des erreurs


# Fonction qui introduit aléatoirement des erreurs dans une image et garde en mémoire la localisation de ces erreurs.
def creation_erreurs(image, nb_erreurs):
    lignes,colonnes = np.shape(image)
    localisation_erreurs = []
    image_alteree = deepcopy(image)

    for i in range(nb_erreurs):
        e_ligne,e_colonne,couleur = randint(0, lignes-1),randint(0, colonnes-1),randint(0, 255)
        localisation_erreurs.append( e_ligne*colonnes + e_colonne )
        image_alteree[e_ligne][e_colonne] = couleur

    return image_alteree,localisation_erreurs


# Fonction qui introduit des erreurs selon une répartition déterminée au préalable.
def creation_erreurs_localisee(image, localisation_erreurs):
    return _creation_erreurs_localises(image, localisation_erreurs, 15)


# Fonction qui introduit au maximum une erreur par blocs de 16 bits.
def creation_erreurs_optimisee(image, erreurs):
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




## Test


def calcul_joconde():
    RESULTS       = []
    HEIGHT        = 417
    WIDTH         = 300
    IMAGE_PATH    = "C:/Users/devan/OneDrive/Bureau/TIPE MP/Image/INPUT/joconde.jpg"
    IMAGE         = plt.imread(IMAGE_PATH)
    IMAGE_GREY    = conversion(IMAGE)
    IMAGE_ENCODED = encodage(IMAGE_GREY)
    NUMBER_OF_ERRORS = [100, 1000, 10000, 25000, 50000]

    for NUMBER in NUMBER_OF_ERRORS:
        IMAGE_ALTERED, ERRORS_LOCATION  = creation_erreurs(IMAGE_GREY, NUMBER)
        IMAGE_ENCODED_AND_ALTERED = creation_erreurs_localisee(IMAGE_ENCODED, ERRORS_LOCATION)
        IMAGE_DECODED = decodage(IMAGE_ENCODED_AND_ALTERED, HEIGHT, WIDTH)
        RESULTS.append([IMAGE_ALTERED, IMAGE_DECODED])

    return RESULTS



def afficher(IMAGES):
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
    afficher(calcul_joconde())






