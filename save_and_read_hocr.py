# zeinab_Taghavi
# saving hOCR on .hocr file:
#   after getting hocr file from pytessract, make a tree from it -> tree
#   then make a file , and store tree as string,
#       why do not store it as string from begin? because in that case we will have '/n's in our file and all hOCR will be stored as a single line
# reading from .hocr file:
#   open .hocr file
#   make a tree from
#   find the pattern you want
import pytesseract
from lxml import etree

# save

f_save = pytesseract.image_to_pdf_or_hocr('image1.bmp', extension='hocr' , lang='fas')
tree = etree.fromstring(f_save)

with open('save_and_read_horc_file.hocr', 'wb') as file:
    file.write(etree.tostring(tree))
    file.close()


# read

f_open = open('save_and_read_horc_file.hocr' , 'r', encoding='iso-8859-1').read().encode('utf-8')
tree_open = etree.fromstring(f_open)
words = tree.xpath("//*[@class='ocr_line']")

#   now we can use it for example finding title inside
for w in words:
    print( w.attrib['title'])

