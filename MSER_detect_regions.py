import cv2
import numpy as np
img = cv2.imread('image1.bmp', 0) # source image

# by this code we will lose picture in eBook , so first,find pictures directions , at last we will put them back
blur = cv2.blur(img , (5,5))
theresh = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)
output_base_image = cv2.bitwise_not(theresh)
contours ,_ = cv2.findContours(output_base_image , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)

images_direction = []

# the image must be between 10% and 90% of main image
max_height = int(round(0.9 * img.shape[0]))
max_width = int(round(0.9 * img.shape[1]))
min_height = int(round(0.1 * img.shape[0]))
min_width = int(round(0.1 * img.shape[1]))

# for any contour in main image , check if it is big enough to ba an image or not
for contour in contours:
    x, y, w, h = cv2.boundingRect(contour)  # position of contour
    if w < max_width and w > min_width and h < max_height and h > min_height \
            and x+w < max_width and  y+h < max_height:
        images_direction.append([x, y, x+w, y+h])



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

new_image = cv2.bitwise_or(final_img, img) # bitwise will clear round of image

# replace picture
for i in images_direction:
    x1, y1, x2, y2 = int(i[0]), int(i[1]), int(i[2]), int(i[3])
    new_image[y1:y2, x1:x2]  = img[y1:y2, x1:x2]


# by this code we will lose picture in eBook , so look at find_picture_in_ebook.py
cv2.imwrite("MSER_detect_regions_on_new_image.jpg",new_image)





