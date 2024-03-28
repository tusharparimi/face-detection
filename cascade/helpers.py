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


def evaluate(preds_val, labels_val):
    tp=np.sum((preds_val==1) & (labels_val==1))
    fn=np.sum((preds_val==-1) & (labels_val==1))
    fp=np.sum((preds_val==1) & (labels_val==-1))
    tn=np.sum((preds_val==-1) & (labels_val==-1))
    return tp/(tp+fn), fp/(fp+tn)



if __name__=="__main__":
    preds_val=np.ones((5,))
    labels_val=np.ones((5,))
    labels_val[0]=-1
    labels_val[1]=-1
    D, F=evaluate(preds_val, labels_val)
    print(D.shape, F.shape)
    print(D)
    print(F)
