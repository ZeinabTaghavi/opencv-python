from lxml import etree
import cv2

dpi = (300,300) # default is (300,300) but you can check with PIL, Image.info
try:
    f = open('z1.hocr', 'r', encoding='iso-8859-1').read().encode('utf-8')  # 'z1.hocr'
except:
    print('hOCR file was not found')

imgTemp = cv2.imread('image1_Modified.jpg') # image file name

tree = etree.fromstring(f)
words = tree.xpath("//*[@class='ocr_line']")

lines_Direction = []

for w in words:
    titles = w.attrib['title'].split()
    x1, y1, x2, y2, x_size = int(titles[1]), int(titles[2]), int(titles[3]), int(titles[4].split(';')[0]), float(
        titles[titles.index('x_size') + 1].split(';')[0])
    font = (x_size * 72) / dpi[0]
    imgTemp = cv2.rectangle(imgTemp , (x1 , y1) , (x2,y2) , (255,0,0) , 3)
    cv2.putText(imgTemp, "font:"+str(int(font)), (x1,y1), cv2.FONT_HERSHEY_SIMPLEX, 1,(0,0,255),2,cv2.LINE_AA)



cv2.imwrite('find_font_size_with_hOCR.jpg',imgTemp)
