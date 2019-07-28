# comparision between finding contours, MSER and hOCR on time they take,
# to find words or approximating to words for Document denoising
# for example : time by using contour: 0.500853192
#               time by using MSER: 3.025797583
#               time by using hOCR: 0.5355071

import cv2
from lxml import etree
img = cv2.imread('image1.bmp') # source image

e1 = cv2.getTickCount()
imgContour = img.copy()
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
blur = cv2.blur(gray , (5,5))
_, thresh = cv2.threshold(img , 127 , 255 , cv2.THRESH_BINARY)
conyours , _ = cv2.findContours(gray , cv2.RETR_TREE , cv2.CHAIN_APPROX_SIMPLE)
max_w = 0.9 * img.shape[0]
max_h = 0.9 * img.shape[1]
min_w = 0.001 * img.shape[0]
min_h = 0.001 * img.shape[1]
for cnt in conyours:
    x , y , w , h = cv2.boundingRect(cnt)
    if w > max_w or w < min_w or h > max_h or h < min_h :
        continue
    imgContour = cv2.rectangle(imgContour , (x,y) , (x+w , y+h) , (255) , 10)
cv2.imwrite('comparision_img_contour.jpg' , imgContour)
e2 = cv2.getTickCount()
print('time by using contour: '+str((e2-e1)/cv2.getTickFrequency()))

e3 = cv2.getTickCount()
imgMSER = img.copy()
mser = cv2.MSER_create()
gray = cv2.cvtColor(img , cv2.COLOR_BGR2GRAY)
regions = mser.detectRegions(gray)
hulls = [cv2.convexHull(p.reshape(-1,1,2)) for p in regions[0]]
cv2.polylines(imgMSER , hulls , True ,255 ,2)
cv2.imwrite('comparision_img_MSER.jpg' , imgMSER)
e4 = cv2.getTickCount()
print('time by using MSER: '+str((e4-e3)/cv2.getTickFrequency()))

e5 = cv2.getTickCount()
imghOSR = img.copy()
f = open('z1.hocr' ,'r', encoding='iso-8859-1').read().encode('utf-8')
tree = etree.fromstring(f)
words = tree.xpath("//*[@class='ocrx_word']")
for w in words:
    title_splited = w.attrib['title'].split()
    x1, y1, x2, y2 = int(title_splited[1]) , int(title_splited[2]) , int(title_splited[3]) , int(title_splited[4].split(';')[0])
    imghOSR = cv2.rectangle(imghOSR , (x1 , y1) , (x2,y2) , (255,0,0) , 3)
cv2.imwrite('comparision_img_hOCR.jpg' , imghOSR)
e6 = cv2.getTickCount()
print('time by using hOCR: '+str((e6-e5)/cv2.getTickFrequency()))

