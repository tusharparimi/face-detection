from myFaceCascade import faceCascade

class faceClassifier:

    def __init__(self, cascade: faceCascade):
        self.cascade=cascade

    def compute(self, win):
        #i=0
        for stage in self.cascade.stages:
            #i=i+1
            #print(i)
            if not stage.classify(win):
                return False
        print("face-detected")
        return True