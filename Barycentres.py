##
from PIL import Image, ImageFont, ImageDraw
import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
from math import *
import scipy
from scipy import interpolate # bibliothèque utilisée pour la dichotomie

# Cette fonction a été paramétrée pour fonctionner avec les fichiers images bmp fournis
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
        # l’algorithme de Scharr a été étudié comme alternative mais avec moins de succès que celui de Sobel
        grad_y = cv.Sobel(gray, ddepth, 0, 1, ksize=3, scale=scale, delta=delta, borderType=cv.BORDER_DEFAULT)

        abs_grad_x = cv.convertScaleAbs(grad_x)
        abs_grad_y = cv.convertScaleAbs(grad_y)

        grad = cv.addWeighted(abs_grad_x, 0.5, abs_grad_y, 0.5, 0)

        #cv.imshow(window_name, grad)
        #cv.waitKey(0)

        return grad

    # chargement de l’image en bmp
    img = cv.imread(path)
    img = img[100:900, 300:1015]# région d’intérêt(affichage de la zone à étudier)
    cv.imwrite("/Users/charlesassoi/Downloads/000000.bmp", img)# chemin servant à stocker une image intermédiaire dans l’étude

    grad=main(["/Users/charlesassoi/Downloads/000000.bmp",])

    # On passe à la détection des blobs après celle des contours

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
    # Il peut être intéressant pour l’étude d’ afficher le nombre de blobs dans«text»
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
    # remarque:les fonctions d’affichages(imshow) fonctionnent comme des return et leur appel nous fait quitter la fonction
    # Pour passer à l'affichage, on peut mettre le return en commentaire temporairement
    return L
    # Show blobs
    cv.imshow("Filtering ellipse Blobs Only", blobs)
    #cv.waitKey(0)
    cv.waitKey(1000) == ord('0')
    cv.destroyAllWindows()

detection_sobel("/Users/charlesassoi/Downloads/Mode 1_2023/002293.bmp")

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
# On teste la fonction liste_tri_haut avec la dection de Sobel (retirer le #)
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



# On cherche ici à afficher les barycentres des blobs
def barycentre(path,ecart_points):
    import scipy
    from scipy import interpolate
    l=liste_tri_haut(path, ecart_points)
    m=liste_tri_bas(path, ecart_points)
    xp = [l[i][0] for i in range (len(l))]
    xp.reverse()
    # xp reçoit les abscisses de notre graphe
    yp = [l[i][1] for i in range (len(l))]
    zp = [m[i][1] for i in range (len(m))]
    zp.reverse()
    yp.reverse()
    # yp reçoit les ordonnées de notre graphe correspondant à ceux de liste_tri_haut
    # zp reçoit les ordonnées de notre graphe correspondant à ceux de liste_tri_bas

    f = scipy.interpolate.interp1d(xp, yp)
    g = scipy.interpolate.interp1d(xp, zp)
    y=f(xp)
    z=g(xp)
    liste=(y-z)/((y-z)[0])
    print(liste)
    im = cv.imread(path)
    im = im[100:900, 300:1015]
    cv.imwrite("/Users/charlesassoi/Downloads/000000.bmp", im)
    #On peut afficher le graphe (retirer le "#" des commentaires ci-dessous
    #im = plt.imread("/Users/charlesassoi/Downloads/111111.bmp")
    #plt.subplots()
    #plt.imshow(im)
    #plt.plot(xp, yp, "b-")
    #plt.plot(xp,zp,"r-")

    #plt.xlabel('coordonnée x')
    #plt.ylabel('coordonnée y')
    #plt.title('Barycentres des mires')
    #plt.grid()
    #plt.show()
    return liste  # Cette liste servira au critère de propagation
# On teste la fonction avec le path du 002295.bmp
barycentre("/Users/charlesassoi/Downloads/Mode 1_2023/002295.bmp",10)