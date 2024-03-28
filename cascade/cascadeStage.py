from stump import decisionStump
from feature import *

class cascadeStage:
    def __init__(self):
        self.stage_id=None
        #self.features=[]
        self.stumps=[] 
        self.stage_threshold=0 

    def add_feature(self, f):
        self.features.append(f)

    def train_stage(self, samples, labels, feature_list, n_stumps):
        n_samples=samples.shape[0]
        w=np.full(n_samples, (1/n_samples))
        self.stumps=[]
        for _ in range(n_stumps):
            stump=decisionStump()
            min_error=float('inf')
            for feature in feature_list:
                feature_values=np.asarray([feature.get_value(x) for x in samples])
                thresholds=np.unique(feature_values)
                for threshold in thresholds:
                    p=1
                    preds=np.ones(n_samples)
                    preds[feature_values<threshold]=-1
                    misclassified=w[labels!=preds]
                    error=sum(misclassified)
                    if error > 0.5:
                        p=-1
                        error=1-error
                    if error < min_error:
                        min_error=error
                        stump.stump_threshold=threshold
                        stump.polarity=p
                        stump.feature=feature
            
            EPS=1e-10
            stump.alpha=0.5*np.log((1.0-min_error+EPS)/(min_error+EPS))
            preds=stump.predict(samples)
            w*=np.exp(-stump.alpha*labels*preds)
            w/=np.sum(w)
            self.stumps.append(stump)
            feature_list.remove(stump.feature)
        self.stage_threshold=sum([stump.alpha for stump in self.stumps])

    def predict(self, samples):
        print(self.stage_threshold)
        stump_preds=[stump.alpha*stump.predict(samples) for stump in self.stumps]
        y_pred=np.sum(stump_preds, axis=0)
        pred=-np.ones(samples.shape[0])
        pred[y_pred >= self.stage_threshold]=1
        return pred
    



if __name__=="__main__":
    samples=np.random.rand(6, 24, 24)
    #samples=np.ones((5, 24, 24))
    samples=np.asarray([integral_image(x) for x in samples])
    labels=np.ones((6))
    labels[2]=-1
    labels[3]=-1
    labels[5]=-1
    #print(labels)
    #print(samples.shape)
    #print(labels.shape)
    feature_list=get_feature_list(24)
    #print(len(feature_list))
    stage=cascadeStage()
    stage.train_stage(samples, labels, feature_list, 3)
    p=np.random.permutation(labels.shape[0])
    s_samples=samples[p]
    s_labels=labels[p]
    print(s_labels)
    preds=stage.predict(s_samples)
    print(preds)
    print(stage.stumps)
    for stump in stage.stumps:
        print(stump.alpha)
    
    
