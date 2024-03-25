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
        self.pos=pos
        self.scale=scale
        self.type=type

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

    def get_value(self):
        print(isinstance(self.type, hort2))
        #TODO: implement get_value by the parameters pos, scale and type


if __name__=="__main__":
    f=feature((0,0), (1, 2), hort2())
    print(f.type.parts_along_h, f.type.parts_along_w)
    f.get_value()
