import numpy as np

class faceFeature:
    def __init__(self):
        pass
    
    def block_sum(self, win, p, lenx, leny):
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

class rect2hort(faceFeature):
    parts_along_h=1
    parts_along_w=2

    def get_features(self, win, lenh, lenw, stride_h=1, stride_w=1):
        feature_dict={}
        (h,w)=win.shape
        for i in range(0,h-lenh+1,stride_h):
            for j in range(0,w-lenw+1,stride_w):
                #print(i, j, lenw, lenh)
                bsum1=self.block_sum(win, (j,i),int(lenw/2),lenh)
                #print("bsum1: ", bsum1)
                bsum2=self.block_sum(win, (j+int(lenw/2),i),int(lenw/2),lenh)
                #print("bsum2: ", bsum2)
                res=bsum2-bsum1
                feature_dict['rect2_hort_'+str(lenh)+'x'+str(lenw)+'_'+'('+str(i)+','+str(j)+')']=res
        return feature_dict

class rect2vert(faceFeature):
    parts_along_h=2
    parts_along_w=1

    def get_features(self, win, lenh, lenw, stride_h=1, stride_w=1):
        feature_dict={}
        (h,w)=win.shape
        for i in range(0,h-lenh+1,stride_h):
            for j in range(0,w-lenw+1,stride_w):
                bsum1=self.block_sum(win, (0,0),lenw,int(lenh/2))
                bsum2=self.block_sum(win, (0,int(lenh/2)),lenw,int(lenh/2))
                res=bsum1-bsum2
                feature_dict['rect2_vert_'+str(lenh)+'x'+str(lenw)+'_'+'('+str(i)+','+str(j)+')']=res
        return feature_dict
    
class rect3hort(faceFeature):
    parts_along_h=1
    parts_along_w=3

    def get_features(self, win, lenh, lenw, stride_h=1, stride_w=1):
        feature_dict={}
        (h,w)=win.shape
        for i in range(0,h-lenh+1,stride_h):
            for j in range(0,w-lenw+1,stride_w):
                bsum1=self.block_sum(win, (0,0),int(lenw/3),lenh)
                bsum2=self.block_sum(win, (int(lenw/3),0),int(lenw/3),lenh)
                bsum3=self.block_sum(win, (int((lenw*2)/3),0),int(lenw/3),lenh)
                res=bsum2-(bsum1+bsum3)
                feature_dict['rect3_hort_'+str(lenh)+'x'+str(lenw)+'_'+'('+str(i)+','+str(j)+')']=res
        return feature_dict
    
class rect4diag(faceFeature):
    parts_along_h=2
    parts_along_w=2

    def get_features(self, win, lenh, lenw, stride_h=1, stride_w=1):
        feature_dict={}
        (h,w)=win.shape
        for i in range(0,h-lenh+1,stride_h):
            for j in range(0,w-lenw+1,stride_w):
                bsum1=self.block_sum(win, (0,0),int(lenw/2),int(lenh/2))
                bsum2=self.block_sum(win, (int(lenw/2),0),int(lenw/2),int(lenh/2))
                bsum3=self.block_sum(win, (0,int(lenh/2)),int(lenw/2),int(lenh/2))
                bsum4=self.block_sum(win, (int(lenw/2),int(lenh/2)),int(lenw/2),int(lenh/2))
                res=bsum2+bsum3-(bsum1+bsum4)
                feature_dict['rect4_diag_'+str(lenh)+'x'+str(lenw)+'_'+'('+str(i)+','+str(j)+')']=res
        return feature_dict
            


if __name__ == "__main__":
        
    # img=np.ones((10,10,3)).astype(np.uint8)
    # detector=faceDetector(img, winSize=3, stride=1)
    # print(detector.int_img.shape)

    # win=detector.int_img[:5, :5]
    # print(win.dtype)
    win=np.ones((5,5), np.int32)
    print(win.shape)
    print(win)

    feature2=faceFeature()
    print(type(feature2))

    feature2=rect2hort()
    print(feature2.get_features(win, 2, 2, 2, 2))
    print(len(feature2.get_features(win, 2, 2, 2, 2)))

    feature2=rect2vert()
    print(feature2.get_features(win, 2, 2, 2, 2))
    print(len(feature2.get_features(win, 2, 2, 2, 2)))

    feature2=rect3hort()
    print(feature2.get_features(win, 3, 3, 2, 2))
    print(len(feature2.get_features(win, 3, 3, 2, 2)))

    feature2=rect4diag()
    print(feature2.get_features(win, 4, 4, 1, 1))
    print(len(feature2.get_features(win, 4, 4, 1, 1)))
    print(type(type(feature2)))

    

