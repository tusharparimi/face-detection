import cv2
import numpy as np
from myFaceClassifier import faceClassifier


class faceDetector:
    def __init__(self, img, winSize, stride):
        assert winSize < img.shape[0]/2 or winSize < img.shape[1]/2, f"winSize {winSize} is not less than 0.5*(image dim)"
        self.winSize=winSize
        self.stride=stride
        self.img=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        self.int_img=self.integral_image()

    def integral_image(self):
        h, w=self.img.shape
        int_img = [ [ 0 for y in range(w) ] for x in range(h) ]
        for y in range(0, h):
                sum = 0
                for x in range(0, w):
                    sum += self.img[y][x]
                    int_img[y][x] = sum
                    if y > 0:
                        int_img[y][x] += int_img[y-1][x]
        return np.asarray(int_img)


    def processImage(self, classifier: faceClassifier):
        rects=[]
        h,w=self.int_img.shape
        for i in range(0, h-self.winSize, self.stride):
            for j in range(0, w-self.winSize, self.stride):
                win=self.int_img[i:i+self.winSize, j:j+self.winSize]
                if classifier.compute(win):
                    rects.append(i, j, self.winSize)
        return rects



if __name__ == "__main__":
    img=cv2.imread("pretrained-cascade\\test.jpg")
    print(img.shape)

    #cv2.imshow("img", img)
    #cv2.waitKey(0)

    detector=faceDetector(img, winSize=100, stride=20)
    #detector.processImage()
