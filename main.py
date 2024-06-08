import cv2
import numpy as np
from matplotlib import pyplot as plt

# Baca gambar dari file
img = cv2.imread('E:\Kuliah\Semester 3\Pengolahan Citra\Codingan\Opencv Segmentasi Plat\plat.jpeg')

# Convert OpenCV Ke matplotlib
img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Buat Gambar
plt.imshow(img_rgb)

# Judul (Opsional)
plt.title('Judul Gambar')

plt.show()