import cv2
import numpy as np
img = cv2.imread('image1.bmp', 0) # source image

first_img = img.copy()
mser = cv2.MSER_create() # use MSER to fine regions

regions = mser.detectRegions(img) # finding regions

hulls = [cv2.convexHull(p.reshape(-1, 1, 2)) for p in regions[0]] # convert them to list that can be used for polygons
cv2.polylines(first_img, hulls, 1, 0) # line polygons around words
cv2.imwrite('MSER_detect_regions_on_source_image.jpg' , first_img)


final_img = np.zeros([img.shape[0], img.shape[1],1],dtype=np.uint8)
final_img.fill(255)


cv2.polylines(final_img, hulls, 1, 0)
cv2.imwrite("MSER_detected_regions.jpg",final_img)

for i in hulls:
    cv2.fillPoly(final_img, [i], 0) # fill the polygon's location with black, then by bitwise can clear round of then

new_image = cv2.bitwise_or(final_img, img) # bitwise wii clear round of image

# by this code we will lose picture in eBook , so look at find_picture_in_ebook.py
cv2.imwrite("MSER_detect_regions_on_new_image.jpg",new_image)





