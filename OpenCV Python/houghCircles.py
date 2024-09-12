import numpy as np
import cv2 as cv

img = cv.imread('testTube.PNG')
img = cv.medianBlur(img, 5)
gray = cv.cvtColor(img, cv.COLOR_BGR2GRAY)

circles = cv.HoughCircles(gray, cv.HOUGH_GRADIENT, 1, 20, param1=50, param2=30, minRadius=0, maxRadius=50)

circles = np.uint16(np.around(circles))

for i in circles[0,:]:
	# draw the outer circles
	cv.circle(img,(i[0],i[1]), i[2],(0,255,0),2)
	# draw the center of the circle
	cv.circle(img,(i[0],i[1]), 2, (0,0,255),3)

cv.imshow('detected circles', img)
cv.waitKey(0)
cv.destroyAllWindows