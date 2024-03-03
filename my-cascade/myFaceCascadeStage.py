import pandas as pd
from myFaceFeatureScale import faceFeatureScale


class faceCascadeStage:
    def __init__(self, model, feature):
        self.model=model
        self.scale_feature=faceFeatureScale(feature)

    def classify(self, win):
        features=self.scale_feature.get_features_all_scales(win)
        df=pd.DataFrame.from_dict([features])
        if self.model.predict(df):
            return True
        return False


      
