import numpy as np


def integral_image(img):
    h, w=img.shape
    int_img = [ [ 0 for y in range(w) ] for x in range(h) ]
    for y in range(0, h):
            sum = 0
            for x in range(0, w):
                sum += img[y][x]
                int_img[y][x] = sum
                if y > 0:
                    int_img[y][x] += int_img[y-1][x]
    return np.asarray(int_img)

# def get_feature_list(winSize, feature_types):
#     feature_list=[]
#     feature_idx=0
#     for ftype in feature_types:
#         ph=ftype.parts_along_h
#         pw=ftype.parts_along_w
#         for lenh in range(ph, winSize+1, ph):
#             for lenw in range(pw, winSize+1, pw):
#                 for p1 in range(0, winSize-lenh+1):
#                     for p0 in range(0, winSize-lenw+1):
#                         #feature_list.append((feature_idx, ftype, (p0,p1), (lenw,lenh))) #, featureType.get_feature_value(win, (p0,p1), lenh, lenw)))
#                         f=feat
#                         feature_list.append()
#                         feature_idx+=1
#     return feature_list