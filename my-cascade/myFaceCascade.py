from myFaceCascadeStage import faceCascadeStage

class faceCascade:
    def __init__(self):
        self.stages=[]
    
    def add_stage(self, stage: faceCascadeStage):
        self.stages.append(stage)

    def get_len(self):
        return len(self.stages)
    
    
    