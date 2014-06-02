import cv2
import numpy as np

cap=cv2.VideoCapture(0)

while(1):
    _, img=cap.read()
    mask = np.zeros(img.shape[:2],np.uint8)
    cv2.imshow('frame',img)
    bgdModel = np.zeros((1,65),np.float64)
    fgdModel = np.zeros((1,65),np.float64)

    rect = (50,50,450,290)
    cv2.grabCut(img,mask,rect,bgdModel,fgdModel,5,cv2.GC_INIT_WITH_RECT)

    mask2 = np.where((mask==2)|(mask==0),0,1).astype('uint8')
    img = img*mask2[:,:,np.newaxis]
    cv2.imshow('detected circles',img)
    k=cv2.waitKey(10)
    if k==ord('q'):
        break

cv2.destroyAllWindows()

