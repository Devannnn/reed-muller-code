"""
joconde        = plt.imread("C:/Users/prigent/Desktop/TIPE/TIPE MP/Image/joconde.jpg")
joconde_grise    = conversion(joconde)
joconde_encode   = encodage(joconde_grise)
joconde_decode0  = decodage(joconde_encode,417,300)

joconde_alt1,l1  = creation_erreurs(joconde_grise,100)
joconde_ealt1    = creation_erreurs_localisee(joconde_encode,l1)
joconde_decode1  = decodage(joconde_ealt1,417,300)

joconde_alt2,l2  = creation_erreurs(joconde_grise,1000)
joconde_ealt2    = creation_erreurs_localisee(joconde_encode,l2)
joconde_decode2  = decodage(joconde_ealt2,417,300)

joconde_alt3,l3  = creation_erreurs(joconde_grise,5000)
joconde_ealt3    = creation_erreurs_localisee(joconde_encode,l3)
joconde_decode3  = decodage(joconde_ealt3,417,300)

joconde_alt4,l4  = creation_erreurs(joconde_grise,10000)
joconde_ealt4    = creation_erreurs_localisee(joconde_encode,l4)
joconde_decode4  = decodage(joconde_ealt4,417,300)

joconde_alt5,l5  = creation_erreurs(joconde_grise,20000)
joconde_ealt5    = creation_erreurs_localisee(joconde_encode,l5)
joconde_decode5  = decodage(joconde_ealt5,417,300)

joconde_alt6,l6  = creation_erreurs(joconde_grise,30000)
joconde_ealt6    = creation_erreurs_localisee(joconde_encode,l6)
joconde_decode6  = decodage(joconde_ealt6,417,300)

joconde_alt7,l7  = creation_erreurs(joconde_grise,35000)
joconde_ealt7    = creation_erreurs_localisee(joconde_encode,l7)
joconde_decode7  = decodage(joconde_ealt7,417,300)

joconde_alt8,l8  = creation_erreurs(joconde_grise,40000)
joconde_ealt8    = creation_erreurs_localisee(joconde_encode,l8)
joconde_decode8  = decodage(joconde_ealt8,417,300)


joconde_alt9,l9  = creation_erreurs(joconde_grise,50000)
joconde_ealt9    = creation_erreurs_localisee(joconde_encode,l9)
joconde_decode9  = decodage(joconde_ealt9,417,300)

joconde_alt10,l10  = creation_erreurs(joconde_grise,60000)
joconde_ealt10     = creation_erreurs_localisee(joconde_encode,l10)
joconde_decode10   = decodage(joconde_ealt10,417,300)

joconde_alt11,l11  = creation_erreurs(joconde_grise,70000)
joconde_ealt11     = creation_erreurs_localisee(joconde_encode,l11)
joconde_decode11   = decodage(joconde_ealt11,417,300)

joconde_alt12,l12  = creation_erreurs(joconde_grise,80000)
joconde_ealt12     = creation_erreurs_localisee(joconde_encode,l12)
joconde_decode12   = decodage(joconde_ealt12,417,300)

joconde_alt13,l13  = creation_erreurs(joconde_grise,90000)
joconde_ealt13     = creation_erreurs_localisee(joconde_encode,l13)
joconde_decode13   = decodage(joconde_ealt13,417,300)

joconde_alt14,l14  = creation_erreurs(joconde_grise,100000)
joconde_ealt14     = creation_erreurs_localisee(joconde_encode,l14)
joconde_decode14   = decodage(joconde_ealt14,417,300)

joconde_alt15,l15  = creation_erreurs(joconde_grise,110000)
joconde_ealt15     = creation_erreurs_localisee(joconde_encode,l15)
joconde_decode15   = decodage(joconde_ealt15,417,300)

joconde_alt16,l16  = creation_erreurs(joconde_grise,120000)
joconde_ealt16     = creation_erreurs_localisee(joconde_encode,l16)
joconde_decode16   = decodage(joconde_ealt16,417,300)

images =   [  joconde_grise, joconde_alt1, joconde_alt2, joconde_alt3,joconde_alt4, joconde_alt5, joconde_alt6,joconde_alt8, joconde_alt9, joconde_alt10, joconde_alt11, joconde_alt12, joconde_alt13,joconde_alt14, joconde_alt15, joconde_alt16,
joconde_decode0, joconde_decode1 , joconde_decode2  , joconde_decode3,
joconde_decode4, joconde_decode5 , joconde_decode6  ,
joconde_decode8, joconde_decode9 , joconde_decode10  , joconde_decode11, joconde_decode12, joconde_decode13,
joconde_decode14, joconde_decode15 , joconde_decode16 ]
"""

def display_images(images):
    lignes , colonnes = 2, 16
    axes=[]
    fig=plt.figure()

    for a in range(lignes*colonnes):
        axes.append( fig.add_subplot(lignes, colonnes, a+1) )
        plt.axis('off')
        plt.title(str(a))
        plt.imshow(images[a], cmap='gray')

    plt.tight_layout()
    plt.show()


def count_errors(image):
    error_count = 0
    ligne,colonne = np.shape(image)

    for i in range(ligne):
        for j in range(colonne):
            if image[i][j] != joconde_grise[i][j]:
                error_count += 1
    return error_count





X  = np.array([100,1000,5000,10000,20000,30000,40000,50000,60000,70000,80000,90000,100000,110000,120000])
Y  = []
Z  = []
for i in range(17,32):
    Y.append(count_errors(images[i]))
    Z.append(417*300)
Y = np.array(Y)
Z = np.array(Z)


plt.plot(X,X/Z,linestyle='-',marker='o',color='b',label="Unencoded image")
plt.plot(X,Y/Z,linestyle='-',marker='o',color='r',label="Encoded then decoded image")
plt.xlabel('Number of introduced errors')
plt.ylabel('Error rate')
plt.legend()
plt.show()














