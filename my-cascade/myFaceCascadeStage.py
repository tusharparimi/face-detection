import numpy as np
from myFaceFeatureNode import faceFeatureNode

class faceCascadeStage:
    def __init__(self):
        self.nodes=[]
        self.parent_idx=
        self.threshold=None
        pass


    def classify(self, wins):
        node_preds=[node.alpha*node.predict(wins) for node in self.nodes]
        y_pred=np.sum(node_preds, axis=0)
        y_pred=np.sign(y_pred)
        return y_pred


      
