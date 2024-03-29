import numpy as np
from cascadeStage import cascadeStage
from stump import decisionStump
from feature import feature, str2ftype
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
        i=0
        F=[None]*(self.num_stages+1)
        D=[None]*(self.num_stages+1)
        n=[0]*(self.num_stages+1)
        F[i]=1.0
        D[i]=1.0
        n[i]=1
        Nset=N[0:P.shape[0]]
        while F[i] > Ftarget:
            samples=np.append(P, Nset, axis=0)
            labels=np.append(np.ones((P.shape[0])), -np.ones((Nset.shape[0])))
            samples, samples_val, labels, labels_val=train_test_split(samples, labels, test_size=0.2, stratify=labels)
            i+=1    #TODO: implement condition if i>num_stages then just break and store the current cascade
            F[i]=F[i-1]
            stage=cascadeStage()
            while F[i] > f*F[i-1]:
                n[i]+=1
                stage.train_stage(samples, labels, feature_list, n[i])
                preds_val=stage.predict(samples_val)
                D[i], F[i]=helpers.evaluate(preds_val, labels_val)
                while D[i] < d*D[i-1]:
                    stage.stage_threshold=stage.stage_threshold-1
                    preds_val=stage.predict(samples_val)
                    D[i], F[i]=helpers.evaluate(preds_val, labels_val)
            self.add_stage(stage)
            stage_tag=ET.SubElement(cascade_tag, "stage")
            stage_tag.set('stage_threshold', str(stage.stage_threshold))
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
                j=0
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
        with open("test_cascade.xml", "wb+") as f:
            f.write(b_xml)

        
    def predict(self, sample):
        for stage in self.stages:
            if stage.predict(sample)[0]==-1:
                return -1
        return 1
    
    def detect(self, img, winSize, stride):
        rects=[]
        int_img=helpers.integral_image(img)
        h, w=int_img.shape
        for i in range(0, h-winSize, stride):
            for j in range(0, w-winSize, stride):
                win=int_img[i:i+winSize, j:j+winSize].reshape(1, winSize, winSize)
                if self.predict(win)==1:
                    rects.append((i, j, winSize))
        return rects    
    

def load_cascade(xml_path):
    tree=ET.parse(xml_path)
    cascade_xml=tree.getroot()
    num_stages=len(cascade_xml)
    cascade_obj=cascade(num_stages)
    for stage_xml in cascade_xml:
        stage=cascadeStage()
        stage.stage_threshold=float(stage_xml.attrib['stage_threshold'])
        for stump_xml in stage_xml:
            stump=decisionStump()
            stump.polarity=int(stump_xml.attrib['polarity'])
            stump.alpha=float(stump_xml.attrib['alpha'])
            stump.stump_threshold=float(stump_xml.attrib['stump_threshold'])
            ftype=str2ftype(stump_xml.attrib['feature_type'])
            pos=helpers.str2tuple(stump_xml.attrib['feature_pos'])
            scale=helpers.str2tuple(stump_xml.attrib['feature_scale'])
            stump.feature=feature(pos, scale, ftype)
            stage.stumps.append(stump)
        cascade_obj.add_stage(stage)
    return cascade_obj


    