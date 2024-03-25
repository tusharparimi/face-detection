from cascadeStage import cascadeStage

class cascade:
    def __init__(self, num_stages):
        self.num_stages=num_stages
        self.stages=[]

    def add_stage(self, stage: cascadeStage):
        self.stages.append(stage)

    def train_cascade(self, P, N, Pval, Nval, Ftarget, f, d):
        i=0
        F=[None]*self.num_stages
        D=[None]*self.num_stages
        n=[0]*self.num_stages
        F[i]=1.0
        D[i]=1.0
        while F[i] > Ftarget:
            i+=1
            F[i]=F[i-1]
            while F[i] > f*F[i-1]:
                stage=cascadeStage()
                stage.train_stage()


        pass
