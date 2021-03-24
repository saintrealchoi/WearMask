import numpy as np
import cv2
import math
from PIL import Image
globrect = 0

def setLabel(img, pts, label):
    # 사각형 좌표 받아오기
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    # cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
    # cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    now = w*h
    return x,y,w,h,pts


image = cv2.imread('C:\\Users\\LG\\Desktop\\card4.jpg')
orig = image.copy()

r = 800.0 / image.shape[0]
dim = (int(image.shape[1] * r), 800)
image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)
org = image.copy()
gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(3,3),0)
edged = cv2.Canny(gray, 75, 200)

(cnts,_) = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
smax = 0
maxX = ()
maxY = ()
maxW = ()
maxH = ()


for pts in cnts:
    # 근사화
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)
    length = cv2.arcLength(pts, True)
    area = cv2.contourArea(pts)

    a,b,c,d,gpt = setLabel(image,pts,'CIR')
    if smax < c*d:
        smax = c*d
        maxX = a
        maxY = b
        maxW = c
        maxH = d
        gpts = gpt


peri = cv2.arcLength(gpts,True)
approx = cv2.approxPolyDP(gpts,0.02*peri,True)

cv2.drawContours(image,[approx],-1,(0,255,0),2)
q,w,e,r = cv2.boundingRect(approx)

cv2.rectangle(image,(q,w),(q+e,w+r),(0,0,255),2)
cv2.rectangle(image, (maxX,maxY), (maxX+maxW,maxY+maxH), (0, 0, 255), 1)

cropped_img = org[w:w+r,q:q+e]
cv2.imshow('croped',cropped_img)
cv2.imwrite("C:\\Users\\LG\\Desktop\\cropped.jpg",cropped_img)
cv2.waitKey(0)
cv2.destroyAllWindows()
