from PIL import Image
import pytesseract
pytesseract.pytesseract.tesseract_cmd = r'C:\\Program Files\\Tesseract-OCR\\tesseract'

text = pytesseract.image_to_string('C:\\Users\\LG\\Desktop\\cropped.jpg',lang='kor')

with open("C:\\Users\\LG\\Desktop\\sample.txt", "w") as f:
    f.write(text)
