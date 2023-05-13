##
from PIL import Image, ImageFont, ImageDraw
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import *
import scipy
from scipy import interpolate

def detection_sobel(path):
    def main(argv):
        window_name = ('Sobel Demo - Simple Edge Detector')
        scale = 1
        delta = 0
        ddepth = cv.CV_16S

        if len(argv) < 1:
            print('Not enough parameters')
            print('Usage:\nmorph_lines_detection.py < path_to_image >')
            return -1
        # Load the image
        src = cv.imread(argv[0], cv.IMREAD_COLOR)
        # Check if image is loaded fine
        if src is None:
            print('Error opening image: ' + argv[0])
            return -1

        src = cv.GaussianBlur(src, (3, 3), 0)

        gray = cv.cvtColor(src, cv.COLOR_BGR2GRAY)

        grad_x = cv.Sobel(gray, ddepth, 1, 0, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)
        # Gradient-Y
        # grad_y = cv.Scharr(gray,ddepth,0,1)
        grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)

        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        #cv.imshow(window_name, grad)
        #cv.waitKey(0)

        return grad

    img = cv.imread(path)
    img = img[100:900, 300:1015]
    cv.imwrite("/Users/charlesassoi/Downloads/000000.bmp", img)
    #img = img[50:230, 50:230]
    #img = cv.resize(img, (500, 1000))
    #cv.imwrite("/Users/charlesassoi/opt/anaconda3/pkgs/imagecodecs-2021.8.26-py39ha952a84_0/info/recipe/mona.jpg", img)
    #grad = main(["/Users/charlesassoi/opt/anaconda3/pkgs/imagecodecs-2021.8.26-py39ha952a84_0/info/recipe/mona.jpg"])
    grad=main(["/Users/charlesassoi/Downloads/000000.bmp",])


    # grad=grad[50:230,50:230]

    # Set our filtering parameters
    # Initialize parameter setting using cv2.SimpleBlobDetector
    params = cv.SimpleBlobDetector_Params()

    # Set Area filtering parameters
    params.filterByArea = True
    params.minArea =5#5*6+1*2

    # Set Circularity filtering parameters
    params.filterByCircularity = True
    params.minCircularity = 0.01#3*6

    # Set Convexity filtering parameters
    params.filterByConvexity = True
    params.minConvexity = 0.1#2*6+1*2

    # Set inertia filtering parameters
    params.filterByInertia = True
    params.minInertiaRatio = 0.01#2*6+1*5

    # Create a detector with the parameters
    detector = cv.SimpleBlobDetector_create(params)

    # Detect blobs
    keypoints = detector.detect(grad)

    # Draw blobs on our image as red circles
    blank = np.zeros((1, 1))
    blobs = cv.drawKeypoints(grad, keypoints, blank, (0, 0, 255), cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    number_of_blobs = len(keypoints)
    #text = "Number of ellipse Blobs: " + str(len(keypoints))
    # cv.putText(blobs, text, (20, 550), cv.FONT_HERSHEY_SIMPLEX, 1, (0, 100, 255), 2)
    #print(text)

    # Recherche des coordonnées et mémorisation des coordonnées dans les listes LX et LY

    LX = []
    LY = []

    for keyPoint in keypoints:
        x = keyPoint.pt[0]
        y = keyPoint.pt[1]
        s = keyPoint.size
        LX.append(x)
        LY.append(y)

    L = np.zeros(shape=(len(LX), 2))

    for i in range(len(LX)):
        L[i][0] = LX[i]
        L[i][1] = LY[i]
    print(L)
    #cv.imwrite("imge.png",blobs)
    return L
    # la fonction retourne L mais pour l'affichage on peut mettre le return temporairement en commentaire et afficher le code ci-dessous
    # Show blobs
    cv.imshow("Filtering ellipse Blobs Only", blobs)
    #cv.waitKey(0)
    cv.waitKey(1000) == ord('0')
    cv.destroyAllWindows()

#detection_sobel("/Users/charlesassoi/Downloads/Mode 1_2023/002295.bmp")

def maxX(L):  # fonction basique pour déterminer une valeur maximale entre plusieurs listes
    m = L[0][0]
    indice = 0
    for i in range(np.shape(L)[0]):
        if L[i][0] > m:
            m = L[i][0]
            indice = i
    return indice  # la fonction retourne l'indice de la coordonnee X maximum dans la liste
def minX(L):  # fonction basique pour déterminer une valeur minimale entre plusieurs listes
    m = L[0][0]
    indice = 0
    for i in range(np.shape(L)[0]):
        if L[i][0] < m:
            m = L[i][0]
            indice = i
    return indice  # la fonction retourne l'indice de la coordonnee X maximum dans la liste

def maxY(L):
    m = L[0][0]
    indice = 0
    for i in range(np.shape(L)[0]):
        if L[0][i] > m:
            m = L[0][i]
            indice = i
    return indice  # la fonction retourne l'indice de la coordonnee Y maximum dans la liste
def liste_tri_haut(path,ecart_points):  # permet de trier par ordre décroissant de coordoonées X les coordonnées des mires supérieures
    L = detection_sobel(path)  # on récupère les coordoonées des mires détectées
    A = int(np.shape(L)[0] / 2)  # on détermine le nombre maximum de mires supérieures
    Lhaut = np.zeros(shape=(A, 2))  # matrice vide à A lignes et 2 colonnes, remplie de 0
    max1 = 0
    max2 = 0
    n = 0
    m = 0

    while np.shape(L)[0] > 1:  # tant que la liste des coordonnées possède des données on réalise la boucle
        max1 = L[maxX(L)][0]  # on récupère la mire avec la coordonnée X la plus grande
        Y1 = L[maxX(L)][1]  # on retient la coordonnée Y associée
        L = np.delete(L, maxX(L), 0)  # on supprime les coordoonées de cette mire de la liste initiale
        max2 = L[maxX(L)][0]  # on cherche la nouvelle coordonnee X max
        Y2 = L[maxX(L)][1]  # on retient la valeur de Y associée
        # print(max1,Y1,max2,Y2)
        if abs(max1 - max2) <= ecart_points / 2:  # on vérifie que les 2 points detectés sont bien sur la même verticale ou presque (couple de points). De cette façon on s'assure que le couple est bien complet. Si ce n'est pas le cas alors la mire seule n'est pas retenue dans la liste triée
            L = np.delete(L, maxX(L), 0)
            m += 1
            if Y1 > Y2:  # dans ce cas c'est une mire supérieure (Y1 au dessus de Y2) alors on peut retenir la mire dans Lhaut
                Lhaut[n][0] = max1
                Lhaut[n][1] = Y1
                n += 1

            else:  # ici Y2 correspond à la mire supérieure donc on ajoute dans Lhaut
                Lhaut[n][0] = max2
                Lhaut[n][1] = Y2
                n += 1

    for i in range(m, A):
        Lhaut = np.delete(Lhaut, m,0)  # on supprime les 0 en trop si jamais on a supprimé certaines mires car le couple n'était pas complet

    return (Lhaut)
#print(liste_tri_haut("/Users/charlesassoi/Downloads/Mode 1_2023/002294.bmp",10))


def liste_tri_bas(path, ecart_points):  # on réalise la même chose pour les mires sur l'éprouvette du bas
    L = detection_sobel(path)
    A = int(np.shape(L)[0] / 2)
    Lbas = np.zeros(shape=(A, 2))
    max1 = 0
    max2 = 0
    n = 0
    m = 0

    while np.shape(L)[0] > 1:
        max1 = L[maxX(L)][0]
        Y1 = L[maxX(L)][1]
        L = np.delete(L, maxX(L), 0)
        max2 = L[maxX(L)][0]
        Y2 = L[maxX(L)][1]
        # print(max1,Y1,max2,Y2)
        if abs(max1 - max2) <= ecart_points / 2:  # on vérifie que les 2 points detectés sont bien sur la même verticale ou presque (couple de points)
            L = np.delete(L, maxX(L), 0)
            m += 1
            if Y1 > Y2:
                Lbas[n][0] = max2
                Lbas[n][1] = Y2
                n += 1

            else:
                Lbas[n][0] = max1
                Lbas[n][1] = Y1
                n += 1

    for i in range(m, A):
        Lbas = np.delete(Lbas, m, 0)

    return (Lbas)




def numerotation_mires(path, destination, ecart_points):
    Lhaut = liste_tri_haut(path, ecart_points)  # on récupère les listres triées
    Lbas = liste_tri_bas(path, ecart_points)
    image = Image.open(path)
    # title_font = ImageFont.truetype('playfair/playfair-font.ttf', 200)
    image_editable = ImageDraw.Draw(image)  # on prépare l'image à l'édition
    u = 0
    for k in range(0, len(Lhaut)):
        title_text_Lhaut = f"{u + 1}B"  # on note le numéro de la mire en précisant si elle sur la partie supérieure ou inférieure
        title_text_Lbas = f"{u + 1}H"
        image_editable.text(tuple(Lhaut[k]), title_text_Lhaut, (255))  # on édite l'image
        image_editable.text(tuple(Lbas[k]), title_text_Lbas, (0))
        u = u + 1

    image.save(destination, "PNG")  # on sauve la nouvelle image avec les numéros dans la destination choisie


#numerotation_mires("/Users/charlesassoi/Downloads/Mode 1_2023/002295.bmp","/Users/charlesassoi/Downloads/111111.bmp", 10)

def barycentre(path,ecart_points):
    import scipy
    from scipy import interpolate
    l=liste_tri_haut(path, ecart_points)
    m=liste_tri_bas(path, ecart_points)
    xp = [l[i][0] for i in range (len(l))]
    xp.reverse()
    yp = [l[i][1] for i in range (len(l))]
    zp = [m[i][1] for i in range (len(m))]
    zp.reverse()
    yp.reverse()
    x = np.linspace(np.min(xp), np.max(xp), len(l))

    f = scipy.interpolate.interp1d(xp, yp)
    g = scipy.interpolate.interp1d(xp, zp)
    y=f(xp)
    z=g(xp)
    liste=(y-z)/((y-z)[0])
    print(liste)
    return liste
    #mettre le return en commentaire pour l'affichage
    im = cv.imread(path)
    im = im[100:900, 300:1015]
    cv.imwrite("/Users/charlesassoi/Downloads/000000.bmp", im)
    im = plt.imread("/Users/charlesassoi/Downloads/000000.bmp")
    plt.subplots()
    plt.imshow(im)
    plt.plot(xp, yp, "y+")
    plt.plot(xp,zp,"y+")
    plt.plot(xp, yp, "b-")
    plt.plot(xp,zp,"r-")

    plt.xlabel('coordonnée x')
    plt.ylabel('coordonnée y')
    plt.title('Barycentres des mires')
    plt.grid()
    plt.show()

#barycentre("/Users/charlesassoi/Downloads/Mode 1_2023/002295.bmp",10)
def fissure(path,ecart_points):
    i = 1
    liste = barycentre(path, ecart_points)
    l = liste_tri_haut(path, ecart_points)
    m = liste_tri_bas(path, ecart_points)
    xp = [l[i][0] for i in range(len(l))]
    xp.reverse()
    yp = [l[i][1] for i in range(len(l))]
    zp = [m[i][1] for i in range(len(m))]
    zp.reverse()
    yp.reverse()
    # ce critère de propagation est plutôt naïf , on pourra modifier le seuil(0.1) ou le decomposer en plusieurs états(dédut delaminage ou delaminage avancé)
    while (abs(liste[i-1]-liste[i])<0.1):
        i+=1

    print("le front de fissure se situe entre la {}èmeet la {}ème mire".format(i, i + 1))
    # Position fissure
    a1 = (yp[i - 1] - zp[i]) / (xp[i - 1] - xp[i])
    a2 = (zp[i - 1] - yp[i]) / (xp[i - 1] - xp[i])
    b1 = yp[i] - a1 * xp[i]
    b2 = zp[i] - a2 * xp[i]
    f1 = lambda x: a1 * x +b1
    f2 = lambda x: a2 * x + b2
    g = lambda x: f2(x) - f1(x)

    # determination point d'intersection
    # méthode de la dichotomie
    def dichotomie(g, a, b, epsilon):
        a, b = xp[i - 1], xp[i]
        # test pour vérifier la présence d'une solution sur [x[i-1], x[i]]
        #if (g(x[i - 1]) * g(x[i]) > 0):
         #   print("rien à afficher")
        while (abs(b - a) > epsilon):
            c = (a + b) / 2
            if (g(a) * g(b) <= 0):
                b = c
            else:
                a = c
        print("l'abscisse du front de fissure dans la région d’interêt est " + str((b + a) / 2) + " et son ordonnée est " + str(f1((b + a) / 2)))
        # On rajoute les offsets pour passer au repère global
        print("l'abscisse du front de fissure dans le repère global est " + str(((b + a) / 2 + 100)) + " et son ordonnée est " + str((f1((b + a) / 2)) + 300))

        return (b + a) / 2
    # 490 est l'abscisse de l'axe des machines de traction dans le repère lié à la zone d'intérêt
    # 100 est le décalage (offset) suivant x dû à la zone d'intérêt
    # 50 correspond à la pré-fissure
    lat = 490 + 100 -50- dichotomie(g, xp[i - 1], xp[i], 0.00001)
    print("la distance laterale de delamination est " + str(lat))

    u = dichotomie(g, xp[i - 1], xp[i], 0.00001)
    i+=1
    # mettre le return en commentaire pour l'affichage
    return ([u, f1(u)])

    liste1=[xp[i-1],xp[i]]
    # Pour l'affichage,on peut temporairement mettre le return en commentaire et activer les commandes ci dessous
    im = cv.imread(path)
    im = im[100:900, 300:1015]
    # on enregistre l’image dans un fichier intermédiaire
    cv.imwrite("/Users/charlesassoi/Downloads/000000.bmp", im)
    plt.subplots()
    plt.imshow(im)

    plt.plot(xp, yp, "b-")
    plt.plot(xp, zp, "r-")
    plt.plot(liste1, [f1(xp[i - 1]),f1(xp[i])], "r-")
    plt.plot(liste1, [f2(xp[i - 1]),f2(xp[i])], "b-")
    plt.plot(u, f1(u), "b+")
    plt.xlabel('coordonnée x')
    plt.ylabel('coordonnée y')
    plt.title('fissure')
    plt.grid()
    plt.show()

#fissure("/Users/charlesassoi/Downloads/Mode 1_2023/002295.bmp",10)
def fissure_serie(path_dossier,ecart_points):
    import os
    from os import listdir
    from os.path import isfile, join
    fichiers = [f for f in listdir(path_dossier) if isfile(join(path_dossier, f))]
    from os import walk
    listeFichiers = []
    for (repertoire, sousRepertoires, fichiers) in walk(path_dossier):
        listeFichiers.extend(fichiers)
    fis = []
    for i in listeFichiers[1:60]:
        fis.append(fissure(path_dossier+"/{}".format(i), ecart_points))
    print(fis)
    print(listeFichiers)
    return(fis)

#fissure_serie("/Users/charlesassoi/Downloads/Mode 1_2023",10)

def ligne_fissure(path_dossier,ecart_points):
    x,y=[],[]
    for L in fissure_serie(path_dossier,ecart_points):
        x.append(L[0])
        y.append(L[1])

    plt.plot(x, y, "r+")
    plt.xlabel('coordonnée x')
    plt.ylabel('coordonnée y')
    plt.title('inter')
    plt.grid()
    plt.show()
#ligne_fissure("/Users/charlesassoi/Downloads/Mode1/Sample1/video_sample_1_s1 copie",10)

def vcct(path,ecart_points):
    F = 45.79#########
    #Area = 45.79*0.01138#####
    Area=0.521
    i = 1
    l = liste_tri_haut(path, ecart_points)
    delta_a=l[0][0]-l[1][0]######
    m = liste_tri_bas(path, ecart_points)
    xp = [l[i][0] for i in range(len(l))]
    xp.reverse()
    yp = [l[i][1] for i in range(len(l))]
    zp = [m[i][1] for i in range(len(m))]
    zp.reverse()
    yp.reverse()
    v=[yp[i]-zp[i] for i in range(len(m))]

    liste1=[(m[i][1]+l[i][1])/2 for i in range(len(m))]
    # Position fissure

    idx=11
    xpp=[]
    d0 = (Area * delta_a) / (v[0] * F)*5
    d=[((Area * delta_a)/(v[i]*F))*5 for i in range(len(m))]
    print(d0)
    print(delta_a)
    print(v[0])
    for i in range(len(m)):
        if(d[i]>=d[i-1]*0.9):
            idx -= 1
            #d = ((Area * delta_a) / (v[i] * F)) * 5

    xpp.append(xp[idx]+d[idx])
    print(idx)
    #mettre return en commentaire pour l'affichage
    return([xp[idx]+d[idx],(yp[idx]+zp[idx])/2])
    #pour l'affichage,mettre le return en commentaire
    im = cv.imread(path)
    im = im[100:900, 300:1015]
    #on retire les # pour afficher la superposition
    #cv.imwrite("/Users/charlesassoi/Downloads/000000.bmp", im)
    #plt.subplots()
    #plt.imshow(im)
    plt.plot(xpp,[(yp[idx]+zp[idx])/2],"y+")
    plt.plot(xp, yp, "b-")
    plt.plot(xp, zp, "r-")
    plt.xlabel('coordonnée x')
    plt.ylabel('coordonnée y')
    plt.title('vcct')
    plt.grid()
    plt.show()

#vcct("/Users/charlesassoi/Downloads/Mode 1_2023/002293.bmp",10)


def fissure_serie_vcct(path_dossier,ecart_points):
    import os
    from os import listdir
    from os.path import isfile, join
    fichiers = [f for f in listdir(path_dossier) if isfile(join(path_dossier, f))]
    from os import walk
    listeFichiers = []
    for (repertoire, sousRepertoires, fichiers) in walk(path_dossier):
        listeFichiers.extend(fichiers)
    fis = []
    for i in listeFichiers[1:61]:#on traite les 60 premiers fichiers
        fis.append(vcct(path_dossier+"/{}".format(i), ecart_points))
    print(fis)
    print(listeFichiers)
    return(fis)

#fissure_serie_vcct("/Users/charlesassoi/Downloads/Mode 1_2023",10)

def ligne_fissure_vcct(path_dossier,ecart_points):
    x,y=[],[]
    for L in fissure_serie_vcct(path_dossier,ecart_points):
        x.append(L[0])
        y.append(L[1])

    plt.plot(x, y, "r+")
    plt.xlabel('coordonnée x')
    plt.ylabel('coordonnée y')
    plt.title('ligne_fissure vcct')
    plt.grid()
    plt.show()
#ligne_fissure_vcct("/Users/charlesassoi/Downloads/Mode 1_2023",10)

