import numpy as np
import cv2 as cv

img = cv.imread('messi.jpg')
b, g, r = cv.split(img)

while True:
	split_image = np.hstack((b, g, r))

	img = cv.merge((b,g,r))

	cv.imshow('image', img)
	if cv.waitKey(1) == ord('q'):
	    	break

cv.destroyAllWindows()