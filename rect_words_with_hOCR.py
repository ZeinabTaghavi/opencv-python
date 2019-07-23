from lxml import etree
import cv2

f = open('z1.hocr' , 'r', encoding='iso-8859-1').read().encode('utf-8')
img = cv2.imread('image1_Modified.jpg') # image file name

tree = etree.fromstring(f)
words = tree.xpath("//*[@class='ocrx_word']")
for w in words:
    title_splited = w.attrib['title'].split()
    x1 , y1 , x2 , y2 = int(title_splited[1]) , int(title_splited[2]) , int(title_splited[3]) , int(title_splited[4].split(';')[0])
    img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 3)


cv2.imwrite('rect_words_with_hOCR.jpg',img)
