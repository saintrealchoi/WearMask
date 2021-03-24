import numpy as np
import cv2
import math

globrect = 0

def setLabel(img, pts, label):
    # 사각형 좌표 받아오기
    (x, y, w, h) = cv2.boundingRect(pts)
    pt1 = (x, y)
    pt2 = (x + w, y + h)
    # cv2.rectangle(img, pt1, pt2, (0, 0, 255), 1)
    # cv2.putText(img, label, pt1, cv2.FONT_HERSHEY_PLAIN, 1, (0, 0, 255))
    now = w*h
    print(now, pt1,pt2)
    return x,y,w,h


image = cv2.imread('C:\\Users\\LG\\Desktop\\card.jpg')
orig = image.copy()

r = 800.0 / image.shape[0]
dim = (int(image.shape[1] * r), 800)
image = cv2.resize(image,dim,interpolation=cv2.INTER_AREA)

gray = cv2.cvtColor(image,cv2.COLOR_BGR2GRAY)
gray = cv2.GaussianBlur(gray,(3,3),0)
edged = cv2.Canny(gray, 75, 200)

# cv2.namedWindow('image', cv2.WINDOW_NORMAL)
# cv2.namedWindow('edged', cv2.WINDOW_NORMAL)
# cv2.imshow("image",image)
# cv2.imshow('edged',edged)
#
# cv2.waitKey(0)

(cnts,_) = cv2.findContours(edged.copy(),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
smax = 0
maxX = ()
maxY = ()
maxW = ()
maxH = ()
for pts in cnts:
    # if cv2.contourArea(pts) < 400:  # 노이즈 제거, 너무 작으면 무시
    #     continue

    # 근사화
    approx = cv2.approxPolyDP(pts, cv2.arcLength(pts, True) * 0.02, True)

    # 근사화 결과 점 갯수
    vtc = len(approx)
    # 3이면 삼각형
    if vtc == 3:
        a,b,c,d = setLabel(image, pts, 'TRI')
    # 4면 사각형
    elif vtc == 4:
        a,b,c,d = setLabel(image, pts, 'RECT')
    else:
        length = cv2.arcLength(pts, True)
        area = cv2.contourArea(pts)
        ratio = 4. * math.pi * area / (length * length)

        if ratio > 0.85:
            a,b,c,d = setLabel(image, pts, 'CIR')
        else:
            a,b,c,d = setLabel(image,pts,'CIR')
    if smax < c*d:
        smax = c*d
        maxX = a
        maxY = b
        maxW = c
        maxH = d


cv2.rectangle(image, (maxX,maxY), (maxX+maxW,maxY+maxH), (0, 0, 255), 1)
print(maxX,maxY)
cv2.imshow('img', image)
cv2.waitKey(0)
cv2.destroyAllWindows()

# # (cnts,_) = cv2.findContours(edged.copy(),cv2.RETR_LIST,cv2.CHAIN_APPROX_SIMPLE)
#
# # print(np.shape(cnts))
# # # print(cnts)
# #
# cnts = sorted(cnts,key = cv2.contourArea, reverse = True)[:5] # [:5]
# # print(cnts[1])
# # # print(cnts[0])
# #
# # num = 0
# # cardimg = np.zeros([100,4,1,2])
# #
# for c in cnts:
#     peri = cv2.arcLength(c,True)
#     approx = cv2.approxPolyDP(c,0.03*peri,True)
#     # print(approx)
#     # print(np.shape(approx))
#     # print(approx)
#     if len(approx) == 4:
#         screenCNt = approx
#         # if num == 4:
#         break
# # print(screenCNt)
# # a = np.zeros((4,1,2))
# # a[0] = [89,326]
# # a[1] = [106,392]
# # a[2] = [140,390]
# # a[3] = [97,326]
# # b = [[89,326],[106,392],[140,390],[97,326]]
# # cv2.rectangle(image,(30,250),(140,390),(0,255,0),3)
# # cv2.drawContours(image,[[89,326],[106,392],[140,390],[97,326]])
# cv2.drawContours(image,[screenCNt],-1,(0,255,0),2)
# cv2.imshow("outline",image)
# #
# cv2.waitKey(0)