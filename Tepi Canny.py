import cv2
import numpy as np
import imutils

image=cv2.imread('E:\Kuliah\Semester 3\Pengolahan Citra\Codingan\Opencv Segmentasi Plat\plat 10.jpeg')

image=imutils.resize(image, width=500)

cv2.imshow("Original Image", image)
cv2.waitKey(0)

gray=cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
cv2.imshow("Grayscale Conversation",gray)
cv2.waitKey(0)

gray=cv2.bilateralFilter(gray, 11, 17, 17)
cv2.imshow("Bilateral Filter", gray)
cv2.waitKey(0)

edged=cv2.Canny(gray, 120, 255)
cv2.imshow("Canny Edges", edged)
cv2.waitKey(0)

# Mengasumsikan 'edged' adalah gambar setelah proses deteksi tepi
(cnts,_)=cv2.findContours(edged.copy(), cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

# Mengurutkan kontur berdasarkan luas area kontur secara menurun
cnts=sorted(cnts,key=cv2.contourArea, reverse = True) [:30] 

# Inisialisasi variabel 'NumberPlateCnt' sebagai None
NumberPlateCnt=None