import numpy as np
from cascadeStage import cascadeStage
import helpers

class cascade:
    def __init__(self, num_stages):
        self.num_stages=num_stages
        self.stages=[]

    def add_stage(self, stage: cascadeStage):
        self.stages.append(stage)


    def train_cascade(self, samples, labels, samples_val, labels_val, Ftarget, f, d, feature_list):
        i=0
        F=[None]*self.num_stages
        D=[None]*self.num_stages
        n=[0]*self.num_stages
        F[i]=1.0
        D[i]=1.0
        n[i]=1
        n_samples_val=samples_val.shape[0]
        while F[i] > Ftarget:
            i+=1
            F[i]=F[i-1]
            stage=cascadeStage()
            while F[i] > f*F[i-1]:
                n[i]+=1
                stage.train_stage(samples, labels, feature_list, n[i])
                #TODO: complete cascade training process
                preds_val=stage.predict(samples_val)
                print(preds_val)
                D[i], F[i]=helpers.evaluate(preds_val, labels_val)
                while D[i] < d*D[i-1]:
                    stage.stage_threshold=stage.stage_threshold-1
                    preds_val=stage.predict(samples_val)
                    D[i], F[i]=helpers.evaluate(preds_val, labels_val)
            if F[i] > Ftarget:
                #TODO: evaluate the current cascade on non-face images and only put false detections in the samples for next stage



        pass
