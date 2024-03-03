import cv2

face_detector=cv2.CascadeClassifier('pretrained-cascade\\haarcascade_frontalface_default.xml')

img=cv2.imread("pretrained-cascade\\test1.jpg")
print(img.shape)
img=cv2.resize(img, None, fx=0.25, fy=0.25, interpolation=cv2.INTER_AREA)
gray=cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
print(gray.shape)


results=face_detector.detectMultiScale(gray, 1.3, 5)
for (x, y, w, h) in results:
    img=cv2.rectangle(img, (x,y), (x+w, y+h), (255, 0, 0), 2)

cv2.imshow("img", img)
cv2.waitKey(0)
cv2.destroyAllWindows()


