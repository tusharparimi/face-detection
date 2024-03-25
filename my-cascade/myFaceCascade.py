from myFaceCascadeStage import faceCascadeStage
from myFaceFeatureBoost import faceFeatureBoost

class faceCascade:
    def __init__(self, num_stages):
        self.num_stages=num_stages
        self.stages=[]
    
    def add_stage(self, stage: faceCascadeStage):
        self.stages.append(stage)

    def get_len(self):
        return len(self.stages)
    
    def train(self, Ftarget, f, d, wins, labels, wins_val, labels_val, scaler):
        i=0
        F=[None]*self.num_stages
        D=[None]*self.num_stages
        n=[0]*self.num_stages
        F[i]=1.0
        D[i]=1.0
        while F[i] > Ftarget:
            i+=1
            n[i]=0
            F[i]=F[i-1]
            while F[i] > f*F[i-1]:
                n[i]+=1
                booster=faceFeatureBoost(n[i])
                booster.train(wins, labels, scaler)
                F[i], D[i]=booster.evaluate(wins_val)
                while D[i] < d*D[i-1]:
                    



            

        pass

    
    
    