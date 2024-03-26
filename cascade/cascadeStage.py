class cascadeStage:
    def __init__(self):
        self.stage_id=None
        self.features=[]
        self.weak_classifiers=[] #TODO: create a weak classifier class
        self.stage_threshold=None 

    def add_feature(self, f):
        self.features.append(f)

    def train_stage(self, P, N, feature_list, num_features):
        #TODO: implement the modified AdaBoost by Viola and Jones with num_features
        pass

