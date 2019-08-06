from lxml import etree
import cv2

img = cv2.imread('image1.bmp')
f = open('z1.hocr' , 'r', encoding='iso-8859-1').read().encode('utf-8')

tree = etree.fromstring(f)
words = tree.xpath("//*[@class='ocrx_word']")
for w in words:
    title_splited = w.attrib['title'].split()
    x1 , y1 , x2 , y2 = int(title_splited[1]) , int(title_splited[2]) , int(title_splited[3]) , int(title_splited[4].split(';')[0])
    img_hocr = cv2.rectangle(img, (x1, y1), (x2, y2), (255,0,0), 3)


# gray2 = cv2.cvtColor(img_hocr , cv2.COLOR_BGR2GRAY)
# final = cv2.bitwise_and(new_image , final_img)
cv2.imwrite('rect_words_with_hOCR.jpg',img_hocr)
