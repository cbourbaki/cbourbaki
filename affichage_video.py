import cv2 as cv
import os
from os import listdir
from os.path import isfile, join
fichiers = [f for f in listdir("/Users/charlesassoi/Downloads/Mode1/Sample1/video_sample_1_s1") if isfile(join("/Users/charlesassoi/Downloads/Mode1/Sample1/video_sample_1_s1", f))]
from os import walk
listeFichiers = []
for (repertoire, sousRepertoires, fichiers) in walk("/Users/charlesassoi/Downloads/Mode1/Sample1/video_sample_1_s1"):
    listeFichiers.extend(fichiers)
#on peut afficher listeFichiers
#print(listeFichiers)

for i in listeFichiers:
    mode=cv.imread("/Users/charlesassoi/Downloads/Mode1/Sample1/video_sample_1_s1/{}".format(i))
    cv.imshow("doc",mode)
    cv.waitKey(1000) == ord('0')
cv.destroyAllWindows()

