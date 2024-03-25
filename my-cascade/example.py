import cv2
import numpy as np
import pickle
from myFaceDetector import faceDetector
from myFaceClassifier import faceClassifier
from myFaceCascade import faceCascade
from myFaceCascadeStage import faceCascadeStage
from myFaceFeatureScale import faceFeatureScaler
from myFaceFeature import *

img=cv2.imread("pretrained-cascade\\test.jpg")
#img=cv2.resize(img, None, fx=0.5, fy=0.5, interpolation=cv2.INTER_AREA)
#img=cv2.imread("NON_facial_images\\15.png")
#img=cv2.resize(img, (24,24), interpolation=cv2.INTER_AREA)
print("Input image shape: ",img.shape)

rect2hort_model=pickle.load(open("tree-modelling\\ab_rect2hort.sav", 'rb'))
rect2vert_model=pickle.load(open("tree-modelling\\ab_rect2vert.sav", 'rb'))
rect3hort_model=pickle.load(open("tree-modelling\\ab_rect3hort.sav", 'rb'))
rect4diag_model=pickle.load(open("tree-modelling\\ab_rect4diag.sav", 'rb'))

feature1=rect2hort()
feature2=rect2vert()
feature3=rect3hort()
feature4=rect4diag()

stage1=faceCascadeStage(rect2hort_model, feature1)
stage2=faceCascadeStage(rect2vert_model, feature2)
stage3=faceCascadeStage(rect3hort_model, feature3)
stage4=faceCascadeStage(rect4diag_model, feature4)

cascade=faceCascade()
cascade.add_stage(stage1)
cascade.add_stage(stage2)
#cascade.add_stage(stage3)
#cascade.add_stage(stage4)
print("No. of stages: ", cascade.get_len())

classifier=faceClassifier(cascade)

detector=faceDetector(img, winSize=24, stride=10)
print("Integral image shape: ", detector.int_img.shape)

rects=detector.processImage(classifier)
print("No. of rectangles: ", len(rects))

for rect in rects:
    rects_img=cv2.rectangle(img, (rect[0], rect[1]), (rect[0]+rect[2], rect[1]+rect[2]), (255, 0, 0), 2)

cv2.imshow("rects_img", rects_img)
cv2.waitKey(0)






