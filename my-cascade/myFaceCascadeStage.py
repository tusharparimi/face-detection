import pandas as pd
from myFaceFeatureScale import faceFeatureScaler


class faceCascadeStage:
    def __init__(self, model, feature):
        self.model=model
        self.scaler=faceFeatureScaler(feature)

    def classify(self, win):
        features=self.scaler.get_features_all_scales(win)
        df=pd.DataFrame.from_dict([features])
        if self.model.predict(df):
            return True
        return False


      
