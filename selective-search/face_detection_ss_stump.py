import cv2
import numpy as np
import time
import random
import os
import pandas as pd
import pickle
import sys
sys.path.insert(1,'C:\\Users\\tusha\\Documents\\projects\\face-detection\\haar-features')
from haar_feature_defs import rect2_haar_hort, rect2_haar_vert, rect3_haar_hort, rect4_haar_diag


img=cv2.imread("selective-search\\test.jpg")
print("img shape: ", img.shape)

ss=cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
ss.setBaseImage(img)

ss.switchToSelectiveSearchFast()
#ss.switchToSelectiveSearchQuality()

start=time.time()
rects=ss.process()
end=time.time()

print(f"time for selective search: {round(end-start, 4)} sec")
print(f"Total regions proposals: {len(rects)}")

res=img.copy()

r2h_model=pickle.load(open("tree-modelling\\ab_rect2hort.sav", 'rb'))
r2v_model=pickle.load(open("tree-modelling\\ab_rect2vert.sav", 'rb'))
r3h_model=pickle.load(open("tree-modelling\\ab_rect3hort.sav", 'rb'))
r4d_model=pickle.load(open("tree-modelling\\ab_rect4diag.sav", 'rb'))

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

    cols=[col for col in df.columns if col.startswith("rect2_hort")]
    X=df[cols]
    print("\n")
    if r2h_model.predict(X):
        print("1")
        cols=[col for col in df.columns if col.startswith("rect2_vert")]
        X=df[cols]
        if r2v_model.predict(X):
            print("2")
            cols=[col for col in df.columns if col.startswith("rect3_hort")]
            X=df[cols]
            if r3h_model.predict(X):
                print("3")
                cols=[col for col in df.columns if col.startswith("rect4_diag")]
                X=df[cols]
                if r4d_model.predict(X):
                    print("4")
                    res=cv2.rectangle(res, (x, y), (x + w, y + h), (0, 0, 255), 1)

    cv2.imshow('res', res)
    k=cv2.waitKey(1)
    if k==ord('q'):
        break
