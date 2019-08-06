# zeinab Taghavi
#
# time: 2.176152333
#
# 1 - make new image
# 2 - create MSER and move the approximate fill poly
# 3 - or (and in inverse:black background) with source

import cv2
import numpy as np

def MSER_move_to_new_image_moved(img): # gray image

    # 1 - make new image

    destination = np.zeros((img.shape[0],img.shape[1]) , np.uint8)
    destination.fill(255)

    # 2 - create MSER and move the approximate fill poly

    mser =cv2.MSER_create()
    regions = mser.detectRegions(img)
    hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]]  # convert them to list that can be used for polygons
    filled_image = cv2.fillPoly(destination, hulls, 0, 0)  # line polygons around words

    # 3 - or (and in inverse:black background) with source

    final = cv2.bitwise_or(img , filled_image)
    cv2.imwrite("MSER_move_to_new_image_moved.jpg" , final)
    return final

if __name__ == "__main__":
    n1 = 1
    n2 = 2
    for i in range(n1,n2):
        e1 = cv2.getTickCount()
        img_file = 'image'+str(i)+'.bmp' # you can change it to your image file
        img = cv2.imread(img_file,0)
        MSER_move_to_new_image_moved(img)
        e2 = cv2.getTickCount()
        print(str(i)+' cost time is:'+str((e2-e1)/cv2.getTickFrequency()))

