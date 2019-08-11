# zeinab Taghavi
# time = between 6 and 9

from lxml import etree
import cv2
import pytesseract

def location_and_word_by_hOCR(file_name):
    image = cv2.imread(file_name)
    h_data = pytesseract.pytesseract.image_to_pdf_or_hocr(image , lang='fas+ara' , extension='hocr')

    tree = etree.fromstring(h_data)

    with open('result_hOCR.hocr' , 'wb') as file_name:
        file_name.write(etree.tostring(tree))

    word_list = tree.xpath('//*[@class="ocrx_word"]')

    location_word_list = []
    for word in word_list:
        titles = word.attrib['title'].split()
        x1, y1, x2, y2 = int(titles[1]), int(titles[2]), int(titles[3]), int(titles[4].split(';')[0])
        temp_line_words = str(word.xpath("string()").encode('utf-8')).split("'") # words of per line are extracted
        location_word_list.append([x1, y1, x2, y2,temp_line_words[1]])

    # print(location_word_list)
    return location_word_list

if __name__ == "__main__":

    file_name = 'image1.bmp'
    e1 = cv2.getTickCount()
    location_and_word_by_hOCR(file_name)
    e2 = cv2.getTickCount()
    print('time: '+str((e2-e1)/cv2.getTickFrequency()))
