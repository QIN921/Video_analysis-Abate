import cv2
import numpy as np
from matplotlib import pyplot as plt

img = cv2.imread('./outline.png')
edges = cv2.Canny(img, 100, 200)
edges2 = cv2.Canny(img, 50, 200)
plt.subplot(131),plt.imshow(img,cmap='gray')
plt.title('Original Image'), plt.xticks([]), plt.yticks([])
plt.subplot(132),plt.imshow(edges,cmap='gray')
plt.title('Edge Image1'), plt.xticks([]), plt.yticks([])
plt.subplot(133),plt.imshow(edges2,cmap='gray')
plt.title('Edge Image2'), plt.xticks([]), plt.yticks([])
plt.show()
