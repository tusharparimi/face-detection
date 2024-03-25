import numpy as np
from myFaceFeature import faceFeature
from myFaceFeatureScale import faceFeatureScaler

class faceFeatureNode():
    def __init__(self, feature_idx, scaler: faceFeatureScaler):
        self.threshold=None
        self.feature_idx=feature_idx
        self.alpha=None
        self.polarity=None
        self.feature_info=scaler.get_feature_info(self.feature_idx)


    def predict(self, wins):
        _, featureType, p, (lenh,lenw)=self.feature_info
        values=np.asarray([featureType.get_feature_value(win, p, lenh, lenw) for win in wins])
        preds=np.ones(values.shape[0])
        if self.polarity==1:
            preds[values < self.threshold]=-1 
        else:
            preds[values > self.threshold]=-1
        
        return preds


