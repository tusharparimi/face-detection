"""
Does the ensemble tree models classify the faces correctly as returned by pre-trained Haar cascade from opencv 
"""


import cv2
import time
import pandas as pd
import pickle
import sys
sys.path.insert(1,'C:\\Users\\tusha\\Documents\\projects\\face-detection\\haar-features')
from haar_feature_defs import rect2_haar_hort, rect2_haar_vert, rect3_haar_hort, rect4_haar_diag

face_detector=cv2.CascadeClassifier('pretrained-cascade\\haarcascade_frontalface_default.xml')
img=cv2.imread("pretrained-cascade\\test1.jpg")
img=cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
res=img.copy()
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

rects=face_detector.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in rects:
    res=cv2.rectangle(res, (x,y), (x+w, y+h), (255, 0, 0), 2)

rf_model=pickle.load(open("tree-modelling\\face_rf.sav", 'rb'))
ab_model=pickle.load(open("tree-modelling\\face_ab.sav", 'rb'))
dt_model=pickle.load(open("tree-modelling\\face_dt.sav", 'rb'))

for i in range(len(rects)):
    x, y, w, h=rects[i]
    roi=img[y:y+h, x:x+w]
    gray=cv2.cvtColor(roi,cv2.COLOR_BGR2GRAY)
    gray24=cv2.resize(gray, (24, 24),interpolation = cv2.INTER_AREA)
    (h24,w24)=gray24.shape
    features={}
    for i in range(1,h24+1):
        for j in range(2,w24+1):
            features.update(rect2_haar_hort(gray24,i,j,i,j))
    
    for i in range(2,h24+1):
        for j in range(1,w24+1):
            features.update(rect2_haar_vert(gray24,i,j,i,j))
    
    for i in range(1,h24+1):
        for j in range(3,w24+1):
            features.update(rect3_haar_hort(gray24,i,j,i,j))
    
    for i in range(2,h24+1):
        for j in range(2,w24+1):
            features.update(rect4_haar_diag(gray24,i,j,i,j))
    
    df=pd.DataFrame.from_dict([features])
    #print(df)
    print(rf_model.predict(df))
    if rf_model.predict(df): #and ab_model.predict(df) and dt_model.predict(df):
        res=cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 2)

    cv2.imshow('res', res)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
