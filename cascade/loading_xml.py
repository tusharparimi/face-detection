from cascade import *
import os
import numpy as np
import os
import cv2
from helpers import integral_image

cascade=load_cascade("test_cascade.xml")

# P_list=os.listdir("facial_images\\")
# P=[]
# for each in P_list[:100]:
#     img=cv2.imread("facial_images\\"+each)
#     img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
#     img=cv2.resize(img, (24,24))
#     P.append(img)
# #P=np.asarray(P)
# P=np.asarray([integral_image(x) for x in P])
# print(P.shape)

# for each in P:
    # print(cascade.predict(each.reshape(1, each.shape[0], each.shape[1])))


img=cv2.imread("pretrained-cascade\\test.jpg")
img=cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(img.shape)

rects=cascade.detect(img, 24, 5)
print(len(rects))

for rect in rects:
    res=cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[2]), (255, 0, 0), 2)

cv2.imshow("res", res)
cv2.waitKey(0)
