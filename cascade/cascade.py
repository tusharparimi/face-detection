import numpy as np
from cascadeStage import cascadeStage
import helpers
from sklearn.model_selection import train_test_split
import xml.etree.ElementTree as ET

class cascade:
    def __init__(self, num_stages):
        self.num_stages=num_stages
        self.stages=[]

    def add_stage(self, stage: cascadeStage):
        self.stages.append(stage)


    def train_cascade(self, P, N, Ftarget, f, d, feature_list):
        cascade_tag=ET.Element('cascade')
        #cascade_tag.set() #TODO: add num_stages attribute to cascade tag in XML
        i=0
        F=[None]*(self.num_stages+1)
        D=[None]*(self.num_stages+1)
        n=[0]*(self.num_stages+1)
        F[i]=1.0
        D[i]=1.0
        n[i]=1
        Nset=N[0:P.shape[0]]
        #print(Nset.shape[0])
        #print(P.shape[0])
        while F[i] > Ftarget:
            samples=np.append(P, Nset, axis=0)
            labels=np.append(np.ones((P.shape[0])), -np.ones((Nset.shape[0])))
            #print(samples.shape)
            #print(labels.shape)
            samples, samples_val, labels, labels_val=train_test_split(samples, labels, test_size=0.2, stratify=labels)
            i+=1
            F[i]=F[i-1]
            stage=cascadeStage()
            while F[i] > f*F[i-1]:
                n[i]+=1
                #print(samples.shape, labels.shape)
                stage.train_stage(samples, labels, feature_list, n[i])
                #TODO: complete cascade training process
                preds_val=stage.predict(samples_val)
                D[i], F[i]=helpers.evaluate(preds_val, labels_val)
                while D[i] < d*D[i-1]:
                    stage.stage_threshold=stage.stage_threshold-1
                    preds_val=stage.predict(samples_val)
                    D[i], F[i]=helpers.evaluate(preds_val, labels_val)
            #print("yes")
            #print(stage)
            self.add_stage(stage)
            stage_tag=ET.SubElement(cascade_tag, "stage")
            stage_tag.set('stage_thereshold', str(stage.stage_threshold))
            for stump in stage.stumps:
                stump_tag=ET.SubElement(stage_tag, "stump")
                stump_tag.set('polarity', str(stump.polarity))
                stump_tag.set('alpha', str(stump.alpha))
                stump_tag.set('stump_threshold', str(stump.stump_threshold))
                stump_tag.set('feature_type', str(stump.feature.type)[9:14])
                stump_tag.set('feature_pos', str(stump.feature.pos))
                stump_tag.set('feature_scale', str(stump.feature.scale))
            print("stage "+str(i)+" added")
            Nset=np.zeros((P.shape))
            if F[i] > Ftarget:
                #TODO: evaluate the current cascade on non-face images and only put false detections in the samples for next stage
                j=0
                print(N.shape)
                for each in N:
                    each=each.reshape(1, each.shape[0], each.shape[1])
                    if self.predict(each)==1:
                        Nset[j]=each
                        j+=1
                    if j==Nset.shape[0]:
                        break
                if j!=Nset.shape[0]:
                    Nset=Nset[0:j]

        ET.indent(cascade_tag)
        b_xml=ET.tostring(cascade_tag)
        #print(type(b_xml))
        with open("test_cascade.xml", "wb+") as f:
            f.write(b_xml)

        

    def predict(self, sample):
        for stage in self.stages:
            if stage.predict(sample)[0]==-1:
                return -1
        return 1
    

    def load_cascade(self, xml):
        #TODO: implement the function for loading the cascade from a XML file
        pass