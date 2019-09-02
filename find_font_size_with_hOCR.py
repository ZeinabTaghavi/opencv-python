import pytesseract
from lxml import etree
import cv2
import numpy as np


imgTemp = cv2.imread('image1.bmp') # image file name

# for better accurancy, its better to have a light threshold first
gray_image = cv2.cvtColor(imgTemp , cv2.COLOR_BGR2GRAY)
blur_image = cv2.blur(gray_image , (5,5))
_,imgHOCR = cv2.threshold(blur_image , 127,255,cv2.THRESH_BINARY)
dpi = (300,300) # default is (300,300) but you can check with PIL, Image.info

try:
    f = pytesseract.pytesseract.image_to_pdf_or_hocr(imgHOCR,lang='fas+ara+eng' , extension='hocr')
except:
    print('hOCR file was not found')


tree = etree.fromstring(f)
words = tree.xpath("//*[@class='ocr_line']")

fonts = []

# if you are using persian or arabic fonts, 
persian_fonts = 1.7
english_font = 1

for w in words:
    titles = w.attrib['title'].split()
    x1, y1, x2, y2, x_size = int(titles[1]), int(titles[2]), int(titles[3]), int(titles[4].split(';')[0]), float(
        titles[titles.index('x_size') + 1].split(';')[0])
    font = (x_size * 72) / (dpi[0]*persian_fonts)
    if (x2-x1)>(y2-y1):
        fonts.append(font)
    imgTemp = cv2.rectangle(imgTemp , (x1 , y1) , (x2,y2) , (255,0,0) , 3)
    cv2.putText(imgTemp, "font:"+str(int(font)), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)


print(np.mean(fonts))
cv2.imwrite('find_font_size_with_hOCR.jpg',imgTemp)
