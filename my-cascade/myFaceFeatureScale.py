import numpy as np
from myFaceFeature import *
#from myFaceDetector import faceDetector


class faceFeatureScale:
    def __init__(self, feature):
        self.feature=feature

    def get_features_all_scales(self, win):
        feature_dict_all_scales={}
        (h,w)=win.shape
        ph=self.feature.parts_along_h
        pw=self.feature.parts_along_w
        for i in range(ph, h+1):
            for j in range(pw, w+1):
                feature_dict_all_scales.update(self.feature.get_features(win,i,j,i,j))
        return feature_dict_all_scales


if __name__=="__main__":
    
    # img=np.ones((10,10,3)).astype(np.uint8)
    # detector=faceDetector(img, winSize=3, stride=1)
    # print(detector.int_img.shape)

    # win=detector.int_img[:5, :5]
    win=np.ones((5,5), np.int32)
    print(win.shape)
    print(win)

    feature=rect2hort()
    print(feature.get_features(win, 2, 2, 2, 2))
    print(len(feature.get_features(win, 2, 2, 2, 2)))

    scale_feature=faceFeatureScale(feature)
    print(scale_feature.get_features_all_scales(win))
    print(len(scale_feature.get_features_all_scales(win)))

    feature=rect3hort()
    print(feature.get_features(win, 3, 3, 3, 3))
    print(len(feature.get_features(win, 3, 3, 3, 3)))

    scale_feature=faceFeatureScale(feature)
    print(scale_feature.get_features_all_scales(win))
    print(len(scale_feature.get_features_all_scales(win)))
