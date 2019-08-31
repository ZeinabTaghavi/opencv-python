import cv2
from matplotlib import pyplot as plt
import numpy as np
img = cv2.imread('image1.bmp')
img = cv2.blur(img , (10,10))
#hist , bins = np.histogram(img.flatten , 256 , [0,256])
plt.hist(img.ravel(),256,[0,200])
plt.show()
print(img.flatten)