import numpy as np
from myFaceFeatureNode import faceFeatureNode
from myFaceFeatureScale import faceFeatureScaler



class faceFeatureBoost:
    def __init__(self, num_nodes):
        self.num_nodes=num_nodes
        self.nodes=[]
        self.booster_threshold=None

    def fit(self, X, y):
        n_samples, n_features=X.shape
        w=np.full(n_samples, 1/n_samples)
        
        self.nodes=[]
        for _ in range(self.num_nodes):
            node=faceFeatureNode()
            min_error=float('inf')
            for feature_idx in range(n_features):
                X_c=X[:, feature_idx]
                thresholds=np.unique(X_c)
                for threshold in thresholds:
                    p=1
                    preds=np.ones(n_samples)
                    preds[X_c < threshold]=-1
                    misclassified=w[y!=preds]
                    error=sum(misclassified)

                    if error>0.5:
                        p=-1
                        error=1-error

                    if error < min_error:
                        min_error=error
                        node.threshold=threshold
                        node.feature_idx=feature_idx
                        node.polarity=p

            EPS=1e-10
            node.alpha=0.5*np.log((1.0-min_error+EPS)/(min_error+EPS))
            preds=node.predict(X)
            w*=np.exp(-node.alpha*y*preds)
            w/=np.sum(w)
            self.nodes.append(node)

    def train(self, wins, labels, scaler: faceFeatureScaler):
        n_samples=wins.shape[0]
        n_features=scaler.num_features
        w=np.full(n_samples, 1/n_samples)
        for _ in range(self.num_nodes):
            min_error=float('inf')
            for feature_idx in range(n_features):
                node=faceFeatureNode(feature_idx, scaler)
                _, featureType, p, (lenh,lenw)=node.feature_info
                values=np.asarray([featureType.get_feature_value(win, p, lenh, lenw) for win in wins])
                thresholds=np.unique(values)
                for threshold in thresholds:
                    p=1
                    preds=np.ones(n_samples)
                    preds[values < threshold]=-1
                    misclassified=w[labels!=preds]
                    error=sum(misclassified)
                    if error > 0.5:
                        p=-1
                        error=1-error
                    if error < min_error:
                        min_error=error
                        node.threshold=threshold
                        node.feature_idx=feature_idx
                        node.polarity=p
            EPS=1e-10
            node.alpha=0.5*np.log((1.0-min_error+EPS)/(min_error+EPS))
            preds=node.predict(wins)
            w*=np.exp(-node.alpha*labels*preds)
            w/=np.sum(w)
            self.nodes.append(node)


    def predict(self, wins):
        node_preds=[node.alpha*node.predict(wins) for node in self.nodes]
        y_pred=np.sum(node_preds, axis=0)
        y_pred=np.sign(y_pred)
        return y_pred
    

    def evaluate(self, wins, labels):
        preds=self.predict(wins)
        tp=np.sum((preds==1) & (labels==1))
        fp=np.sum((preds==1) & (labels==-1))
        fn=np.sum((preds==-1) & (labels==1))
        tn=np.sum((preds==-1) & (labels==-1))
        totalp=tp+fn
        if fp+tn > 0: fp_rate=fp/(fp+tn)
        else: fp_rate=0.0
        if totalp > 0: d_rate=tp/totalp
        else: d_rate=0.0
        return fp_rate, d_rate
    
