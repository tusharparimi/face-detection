import numpy as np
from helpers import integral_image

class featureType:
    def __init__(self):
        self.parts_along_h=None
        self.parts_along_w=None

    def set_parts(self, parts_along_h, parts_along_w):
        self.parts_along_h=parts_along_h
        self.parts_along_w=parts_along_w

class hort2(featureType):
    def __init__(self):
        self.set_parts()

    def set_parts(self):
        self.parts_along_h=1
        self.parts_along_w=2

class hort3(featureType):
    def __init__(self):
        self.set_parts()

    def set_parts(self):
        self.parts_along_h=1
        self.parts_along_w=3

class vert2(featureType):
    def __init__(self):
        self.set_parts()

    def set_parts(self):
        self.parts_along_h=2
        self.parts_along_w=1

class vert3(featureType):
    def __init__(self):
        self.set_parts()

    def set_parts(self):
        self.parts_along_h=3
        self.parts_along_w=1

class diag4(featureType):
    def __init__(self):
        self.set_parts()

    def set_parts(self):
        self.parts_along_h=2
        self.parts_along_w=2


class feature:
    def __init__(self, pos, scale, type):
        self.type=type
        self.pos=pos      #pos=(p[0], p[1])
        assert scale[0]>=self.type.parts_along_w and scale[0]%self.type.parts_along_w==0, f"scale of w not enough to accomodate parts of feature along w"
        assert scale[1]>=self.type.parts_along_h and scale[1]%self.type.parts_along_h==0, f"scale of h not enough to accomodate parts of feature along h"
        self.scale=scale  #scale=(lenw, lenh)

    def block_sum(self, win, p, lenx, leny):
        #print(win.shape)
        h, w=win.shape
        assert p[0]+lenx<=w and p[1]+leny<=h, f"point {p} and length combo {lenx, leny} out of bound"

        bsum=win[p[1]+leny-1][p[0]+lenx-1]
        if p[1]-1>=0 and p[0]-1>=0:
            bsum+=win[p[1]-1][p[0]-1]
        if p[1]-1>=0:
            bsum+=-win[p[1]-1][p[0]+lenx-1]
        if p[0]-1>=0:
            bsum+=-win[p[1]+leny-1][p[0]-1]
        return bsum

    def get_value(self, win):
        if isinstance(self.type, hort2):
            bsum1=self.block_sum(win, (self.pos[0],self.pos[1]),int(self.scale[0]/2),self.scale[1])
            bsum2=self.block_sum(win, (self.pos[0]+int(self.scale[0]/2),self.pos[1]),int(self.scale[0]/2),self.scale[1])
            res=bsum2-bsum1
            return res
        if isinstance(self.type, vert2):
            bsum1=self.block_sum(win, (self.pos[0],self.pos[1]),self.scale[0],int(self.scale[1]/2))
            bsum2=self.block_sum(win, (self.pos[0],self.pos[1]+int(self.scale[1]/2)),self.scale[0],int(self.scale[1]/2))
            res=bsum1-bsum2
            return res
        if isinstance(self.type, hort3):
            bsum1=self.block_sum(win, (self.pos[0],self.pos[1]),int(self.scale[0]/3),self.scale[1])
            bsum2=self.block_sum(win, (self.pos[0]+int(self.scale[0]/3),self.pos[1]),int(self.scale[0]/3),self.scale[1])
            bsum3=self.block_sum(win, (self.pos[0]+int((self.scale[0]*2)/3),self.pos[1]),int(self.scale[0]/3),self.scale[1])
            res=bsum2-(bsum1+bsum3)
            return res
        if isinstance(self.type, vert3):
            bsum1=self.block_sum(win, (self.pos[0],self.pos[1]),self.scale[0],int(self.scale[1]/3))
            bsum2=self.block_sum(win, (self.pos[0],self.pos[1]+int(self.scale[1]/3)),self.scale[0],int(self.scale[1]/3))
            bsum3=self.block_sum(win, (self.pos[0],self.pos[1]+int((self.scale[1]*2)/3)),self.scale[0],int(self.scale[1]/3))
            res=bsum2-(bsum1+bsum3)
            return res
        if isinstance(self.type, diag4):
            bsum1=self.block_sum(win, (self.pos[0],self.pos[1]),int(self.scale[0]/2),int(self.scale[1]/2))
            bsum2=self.block_sum(win, (self.pos[0]+int(self.scale[0]/2),self.pos[1]),int(self.scale[0]/2),int(self.scale[1]/2))
            bsum3=self.block_sum(win, (self.pos[0],self.pos[1]+int(self.scale[1]/2)),int(self.scale[0]/2),int(self.scale[1]/2))
            bsum4=self.block_sum(win, (self.pos[0]+int(self.scale[0]/2),self.pos[1]+int(self.scale[1]/2)),int(self.scale[0]/2),int(self.scale[1]/2))
            res=bsum2+bsum3-(bsum1+bsum4)
            return res
        

def get_feature_list(winSize, feature_types=[hort2(), vert2(), hort3(), vert3(), diag4()]):
    feature_list=[]
    feature_idx=0
    for ftype in feature_types:
        ph=ftype.parts_along_h
        pw=ftype.parts_along_w
        for lenh in range(ph, winSize+1, ph):
            for lenw in range(pw, winSize+1, pw):
                for p1 in range(0, winSize-lenh+1):
                    for p0 in range(0, winSize-lenw+1):
                        #feature_list.append((feature_idx, ftype, (p0,p1), (lenw,lenh))) #, featureType.get_feature_value(win, (p0,p1), lenh, lenw)))
                        f=feature((p0, p1), (lenw, lenh), ftype)
                        feature_list.append(f)
                        feature_idx+=1
    return feature_list


def str2ftype(s):
    if s=="hort2":
        ftype=hort2()
    elif s=="hort3":
        ftype=hort3()
    elif s=="vert2":
        ftype=vert2()
    elif s=="vert3":
        ftype=vert3()
    else:
        ftype=diag4()
    return ftype
    



if __name__=="__main__":
    f=feature((1,1), (2, 6), vert3())
    print(f.type, f.pos, f.scale)
    print(f.type.parts_along_h, f.type.parts_along_w)
    img=np.ones((8, 8))
    win=integral_image(img)
    print(win.shape)
    print(win)
    value=f.get_value(win)
    print(value)
