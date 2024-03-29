from cascade import *
import os
import numpy as np
import os
import cv2
from helpers import integral_image

cascade=load_cascade("test_cascade.xml")

P_list=os.listdir("facial_images\\")
P=[]
for each in P_list[:100]:
    img=cv2.imread("facial_images\\"+each)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img, (24,24))
    P.append(img)
#P=np.asarray(P)
P=np.asarray([integral_image(x) for x in P])
print(P.shape)

for each in P:
    print(cascade.predict(each.reshape(1, each.shape[0], each.shape[1])))
