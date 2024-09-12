import cv2 as cv
import numpy as np

#frame = cv.imread('testTube.PNG')

#resize_frame = cv.resize(frame, (400, 800))
resize_frame = cv.imread('testTube.PNG')

gray = cv.cvtColor(resize_frame, cv.COLOR_BGR2GRAY)

sift = cv.SIFT_create()
kp = sift.detect(gray,None)

img = cv.drawKeypoints(gray,kp,resize_frame, (255, 0, 0), 
                                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

spl_img = np.hstack((resize_frame, img))

cv.imshow('frame', spl_img)
    
cv.waitKey(0)

cv.destroyAllWindows()