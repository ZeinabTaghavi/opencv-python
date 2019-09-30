import numpy as np
import cv2
from skimage.data import page
from skimage.filters import (threshold_otsu, threshold_niblack,threshold_sauvola)

img = cv2.imread('img.jpg',0)
blur = cv2.GaussianBlur(img,(5,5),0)
ret, otsu = cv2.threshold(blur,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
cv2.imwrite('{}_otsu.jpg'.format(str(0)),otsu)

thresh = threshold_niblack(img, window_size=5, k=1)
cv2.imwrite('{}_niblack.jpg'.format(str(0)),thresh)

thresh = threshold_sauvola(img, window_size=3)
cv2.imwrite('{}_sauvola.jpg'.format(str(0)),thresh)

