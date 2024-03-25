
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

    def get_feature_value(self, win, p, lenh, lenw):
        bsum1=self.block_sum(win, (p[0],p[1]),int(lenw/2),lenh)
        bsum2=self.block_sum(win, (p[0]+int(lenw/2),p[1]),int(lenw/2),lenh)
        res=bsum2-bsum1
        return res

class rect2vert(faceFeature):
    parts_along_h=2
    parts_along_w=1

    def get_feature_value(self, win, p, lenh, lenw):
        bsum1=self.block_sum(win, (p[0],p[1]),lenw,int(lenh/2))
        bsum2=self.block_sum(win, (p[0],p[1]+int(lenh/2)),lenw,int(lenh/2))
        res=bsum1-bsum2
        return res
    
class rect3hort(faceFeature):
    parts_along_h=1
    parts_along_w=3

    def get_feature_value(self, win, p, lenh, lenw):
        bsum1=self.block_sum(win, (p[0],p[1]),int(lenw/3),lenh)
        bsum2=self.block_sum(win, (p[0]+int(lenw/3),p[1]),int(lenw/3),lenh)
        bsum3=self.block_sum(win, (p[0]+int((lenw*2)/3),p[1]),int(lenw/3),lenh)
        res=bsum2-(bsum1+bsum3)
        return res
    
class rect3vert(faceFeature):
    parts_along_h=3
    parts_along_w=1

    def get_feature_value(self, win, p, lenh, lenw):
        bsum1=self.block_sum(win, (p[0],p[1]),lenw,int(lenh/3))
        bsum2=self.block_sum(win, (p[0],p[1]+int(lenh/3)),lenw,int(lenh/3))
        bsum3=self.block_sum(win, (p[0],p[1]+int((lenh*2)/3)),lenw,int(lenh/3))
        res=bsum2-(bsum1+bsum3)
        return res
    
class rect4diag(faceFeature):
    parts_along_h=2
    parts_along_w=2

    def get_feature_value(self, win, p, lenh, lenw):
        bsum1=self.block_sum(win, (p[0],p[1]),int(lenw/2),int(lenh/2))
        bsum2=self.block_sum(win, (p[0]+int(lenw/2),p[1]),int(lenw/2),int(lenh/2))
        bsum3=self.block_sum(win, (p[0],p[1]+int(lenh/2)),int(lenw/2),int(lenh/2))
        bsum4=self.block_sum(win, (p[0]+int(lenw/2),p[1]+int(lenh/2)),int(lenw/2),int(lenh/2))
        res=bsum2+bsum3-(bsum1+bsum4)
        return res
            


