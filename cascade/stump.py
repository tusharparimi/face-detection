import numpy as np

class decisionStump:
    def __init__(self):
        self.polarity=1
        self.stump_threshold=None
        self.feature=None
        self.alpha=None

    def predict(self, samples):
        n_samples=samples.shape[0]
        feature_values=np.asarray([self.feature.get_value(x) for x in samples])
        #feature_values=np.array([self.feature.get_value(samples[])])
        preds=np.ones(n_samples)
        if self.polarity==1:
            preds[feature_values<self.stump_threshold]=-1
        else:
            preds[feature_values>self.stump_threshold]=-1
        return preds