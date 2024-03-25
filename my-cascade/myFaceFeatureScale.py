import numpy as np
from myFaceFeature import *
#from myFaceDetector import faceDetector


class faceFeatureScaler:
    def __init__(self, winSize, featureTypes):
        self.winSize=winSize
        self.featureTypes=featureTypes
        self.all_features=self.get_all_features()
        self.num_features=len(self.all_features)

    def get_all_features(self):
        all_features=[]
        feature_idx=0
        for featureType in self.featureTypes:
            ph=featureType.parts_along_h
            pw=featureType.parts_along_w
            for lenh in range(ph, self.winSize+1, ph):
                for lenw in range(pw, self.winSize+1, pw):
                    for p1 in range(0, self.winSize-lenh+1):
                        for p0 in range(0, self.winSize-lenw+1):
                            all_features.append((feature_idx, featureType, (p0,p1), (lenh,lenw))) #, featureType.get_feature_value(win, (p0,p1), lenh, lenw)))
                            feature_idx+=1
        return all_features
    
    def get_feature_info(self, feature_idx):
        feature_info=self.all_features[feature_idx]
        return feature_info
        

if __name__=="__main__":
    
    win=np.ones((24,24))
    featureTypes=[rect2hort(), rect2vert(), rect3hort(), rect3vert(), rect4diag()]
    scaler=faceFeatureScaler(win.shape[0], featureTypes)
    print(scaler.get_feature_info(1))
    
    