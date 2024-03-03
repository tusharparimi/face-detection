
class faceCascade:
    def __init__(self):
        self.stages=[]
    
    def add_stage(self, stage):
        self.stages.append(stage)

    def get_len(self):
        return len(self.stages)
    


class faceCascadeStage:
    def __init__(self, model, feature):
        self.model=model
        self.feature=feature

    def classify(self, win):
        df=self.feature.get_features(win)
        if self.model.predict(df):
            return True
        return False
      
