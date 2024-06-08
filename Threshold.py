import cv2
import numpy as np
import imutils

image=cv2.imread('E:\Kuliah\Semester 3\Pengolahan Citra\Codingan\Opencv Segmentasi Plat\plat 10.jpeg')

image=imutils.resize(image, width=500)

cv2.imshow('Original Image', image)
cv2.waitKey(0)

gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Conversation",gray)
cv2.waitKey(0)

gray=cv2.bilateralFilter(gray, 11, 17, 17)
#gray=cv2.GaussianBlur(gray,(5,5),0)
cv2.imshow("Bilateral Filter", gray)
cv2.waitKey(0)

rev, thresh1=cv2.threshold(gray, 120, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imshow('Otsu Threshold',thresh1)
cv2.waitKey(0)

# Mengasumsikan 'thresh1' adalah gambar biner setelah proses thresholding
(cnts,_)=cv2.findContours(thresh1.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Mengurutkan kontur berdasarkan luas area kontur secara menurun
cnts=sorted(cnts,key=cv2.contourArea, reverse = True) [:30] 

# Inisialisasi variabel 'NumberPlateCnt' sebagai None
NumberPlateCnt=None