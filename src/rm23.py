## MODULES
import numpy as np
from random import randint
import matplotlib.pyplot as plt
from pathlib import Path
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


## DECODAGE


# Fonction qui détermine les vecteurs caractéristiques d'un monôme de degré 2 dont la ligne est la ligne i.
def vecteur_caracteristique_degre_2(r,m,i):
    RM = matrice_generatrice_RM(r,m)
    v_caracteristique = RM [len(RM)-i]
    return [v_caracteristique,complementaire(v_caracteristique)]


def couples(m):
    P=[]
    for i in range(0,2*m-2,m-1):
        for j in range(1,2*m-2):
            if j!=i+m-1 and i!=j :
                P.append((i,j))
    return P


# Fonction qui détermine les vecteurs caractéristiques d'un monôme de degré 1 dont la ligne est la ligne i.
def vecteur_caracteristique_degre_1(r,m,i):
    L=[]
    v = hyperplans(m)
    caracteristique = []
    couples_v = couples(m)

    #sépare les vecteurs v_j avec i!=j
    for j in range(1,len(v)):
        if j!=i:
            L.append(v[j])

    #ajout du complémentaire
    for k in range(len(L)):
        L.append(complementaire(L[k]))

    #on multiplie les couples, qui nous intéressent, qui sont déterminés par la fonction couples().
    L = np.array(L)
    for X in couples_v:
        a,b = X
        caracteristique.append(L[a]*L[b])
    return caracteristique


def couples2():
    L = []
    for i in range(2):
        for j in range(2,4):
            for h in range(4,6):
                L.append((i,j,h))
    return L


# Fonction qui détermine les vecteurs caractéristiques du monôme de degré 0 correspondant à la première ligne de la matrice.
def vecteur_caracteristique_degre_0(r,m):
    v = hyperplans(m)[1:]
    L = []
    caracteristique = []
    couples_v = couples2()

    for i in range(m):
        L.append(v[i])
        L.append(complementaire(v[i]))

    #on multiplie les couples, qui nous intéressent, qui sont déterminés par la fonction couples2().
    L = np.array(L)
    for X in couples_v:
        a,b,c = X
        caracteristique.append(L[a]*L[b]*L[c])
    return caracteristique


# Fonction qui créée la liste complète des vecteurs caractéristiques de chaque monôme de la matrice, elle est appelée une et une seule fois d'où l'utilisation d'un tuple pour assurer l'intégrité des données pendant l'exécution du programme.
def creation_vecteur_caracteristique(r,m):
    vecteur_caracteristique = [vecteur_caracteristique_degre_0(2,3)]
    for i in range(1,m):
        vecteur_caracteristique.append(vecteur_caracteristique_degre_1(2,3,i))
    for j in range(m,7):
        vecteur_caracteristique.append(vecteur_caracteristique_degre_2(2,3,j))
    return vecteur_caracteristique

vecteur_caracteristique = tuple(creation_vecteur_caracteristique(2,3))

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

# Fonction qui décode, avec la méthode du vote majoritaire, un vecteur préalablement encodé par la matrice génératrice du code RM(r,m), ici il n'est efficace que pour le code RM(2,3).
def decodage_vecteur(r,m,message):
    RM = matrice_generatrice_RM(r,m)
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
    S2 = produit_vecteur_matrice(message_decode[1:4],RM[1:4])
    E2 = addition_mod2(S2,E1)

    # Calcul des monômes de degre 0
    message_decode[0] = vote_majoritaire(r,m,0,E2)
    return message_decode


## Encodage du message

# Fonction qui convertit une image, i.e , une matrice de pixels en une liste de liste de bits.
def conv_pixels(image):
    # on sépare les pixels et on en fait une liste
    L=[]
    for X in image:
        for Y in  X:
            # chaque pixel est codé sur 8 bits
            L.append(conv_base_2(8,Y))
    # L est une liste de listes de 8 bits or nous travaillons avec des objets de 7 bits donc nous séparons chaque pixel qui seront assemblés en liste de 7 bits dans la fonction decoupage.
    H = []
    for X in L:
        for Y in  X:
            H.append(Y)
    return H


# Fonction qui transforme une image en une liste de liste de 7 bits qui pourront ensuite être encodées.
def decoupage(image):
    L = []
    copie = conv_pixels(image)
    taille = len(copie)
    for k in range(0,taille,7):
            L.append(copie[k:k+7])
    return L


# Fonction qui encode une image en traitant chaque liste de 7 bits.
def encodage(image):
    liste_pixels = decoupage(image)
    RM = matrice_generatrice_RM(2,3)
    for i in range(len(liste_pixels)-1):
        liste_pixels[i] = produit_vecteur_matrice(liste_pixels[i],RM)
    return liste_pixels


# Fonction réciproque de decoupage, elle prend en entrée une liste de vecteurs qui correspondent aux listes de 7 bits, précédemment citées, qui ont été décodées.
def recollage(image,hauteur,largeur):
    copie,L = aplatir(image), []

    # reconversion de chaque liste de 7 bits en l'entier correspondant
    for k in range(0,len(copie),8):
        L.append(base10(copie[k:k+8]))

    matrice = np.zeros((hauteur,largeur), dtype=np.uint8 )
    indice_ligne = 0

    # restauration de la matrice de départ qui représente l'image.
    for x in range(0,len(L)):
        matrice[indice_ligne][x%largeur] = L[x]
        if x!=0 and not(x%largeur):
            indice_ligne +=1
    return matrice


# Fonction réciproque de encodage, elle prend en entrée une liste de vecteurs de 7 bits.
def decodage(image,hauteur,largeur):
    image_decode  = []

    # chaque vecteur est décodé grâce au code RM(2,3)
    for X in image:
        if len(X)==8:
            image_decode.append(decodage_vecteur(2,3,X))
        else:
            image_decode.append(X)

    # l'image est reformée
    image_decode = recollage(image_decode,hauteur,largeur)
    return np.array(image_decode)



## Introduction d'erreurs

# Fonction qui introduit aléatoirement des erreurs dans une image et garde en mémoire la localisation de ces erreurs.
def creation_erreurs(image,nb_erreurs):
    lignes,colonnes = np.shape(image)
    localisation_erreurs = []
    image_alteree = np.copy(image)

    for i in range(nb_erreurs):
        e_ligne,e_colonne,couleur = randint(0, lignes-1),randint(0, colonnes-1),randint(0, 255)
        localisation_erreurs.append(    e_ligne*colonnes + (e_colonne-1)  )
        image_alteree[e_ligne][e_colonne] = couleur

    return image_alteree,localisation_erreurs

# Fonction qui introduit des erreurs selon une répartition déterminée au préalable.
def creation_erreurs_localises(image,localisation_erreurs):
    return _creation_erreurs_localises(image, localisation_erreurs, 7)


## Test de l'encodage et du décodage

RM = matrice_generatrice_RM(2,3)

v1=np.array([1, 1, 1, 1, 0, 0, 0 ])
v2=np.array([1, 0, 1, 0, 1, 0, 1 ])

message = np.array([0,1,1,1,0,1,0])
message_encode = produit_vecteur_matrice(message,RM)
e1 = vote_majoritaire(2,3,1,message_encode)
d1 = decodage_vecteur(2,3,message_encode)


message2 = np.array([0, 0, 1, 0, 0, 0, 1])
message_encode2 = produit_vecteur_matrice(message2,RM)
e2 = vote_majoritaire(2,3,2,message_encode2)
d2 = decodage_vecteur(2,3,message_encode2)

test =  np.array([[100,105,200],[45,204,0],[94,0,204]])

a = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 0, 0, 0]])

b = np.array([[1, 1, 0, 0, 0, 0, 0, 0],
              [1, 0, 1, 0, 0, 0, 0, 0],
              [1, 0, 0, 0, 1, 0, 0, 0]])


res1 = decoupage(a)
res2 = encodage(res1)
res3 = []

for X in res2:
    if len(X)==8:
        res3.append(decodage_vecteur(2,3,X))
    else:
        res3.append(X)


## Test de la génération des erreurs


DATA_DIR = Path(__file__).resolve().parent.parent / "data"
surfeur          = plt.imread(DATA_DIR / "surfer.jpg")
surfeur_gris     = conversion(surfeur)


def calcul():
    surfeur_encode = encodage(surfeur_gris)
    surfeur_decode0 = decodage(surfeur_encode,427,640)


    surfeur_alt1,l1  = creation_erreurs(surfeur_gris,100)
    surfeur_alt2,l2  = creation_erreurs(surfeur_gris,1000)

    surfeur_ealt1    = creation_erreurs_localises(surfeur_encode,l1)
    surfeur_ealt2    = creation_erreurs_localises(surfeur_encode,l2)


    surfeur_decode1 = decodage(surfeur_ealt1,427,640)
    surfeur_decode2 = decodage(surfeur_ealt2,427,640)


    return [ surfeur_gris    , surfeur_alt1    , surfeur_alt2    ,
            surfeur_decode0 , surfeur_decode1 , surfeur_decode2 ]


def afficher(images):
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
    afficher(calcul())







