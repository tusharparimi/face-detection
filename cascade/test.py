import numpy as np
from cascade import cascade
from feature import get_feature_list
import os
import cv2
from helpers import integral_image

cascade=cascade(num_stages=3)
# P=np.random.rand(5, 24, 24)
# N=np.random.rand(15, 24, 24)
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


N_list=os.listdir("NON_facial_images\\")
N=[]
for each in N_list[:300]:
    img=cv2.imread("NON_facial_images\\"+each)
    img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    img=cv2.resize(img, (24,24))
    N.append(img)
#N=np.asarray(N)
N=np.asarray([integral_image(x) for x in N])
print(N.shape)

#print(P.shape[0], N.shape[0])
Ftarget=0.01
f=0.5
d=0.9
feature_list=get_feature_list(24)

print("\ntraining...")
cascade.train_cascade(P, N, Ftarget, f, d, feature_list)
