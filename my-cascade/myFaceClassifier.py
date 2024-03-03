class faceClassifier:

    def __init__(self, cascade):
        self.cascade=cascade

    def compute(self, win):
        for stage in self.cascade.stages:
            if not stage.classify(win):
                return False
        return True