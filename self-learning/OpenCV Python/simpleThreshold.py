import cv2 as cv
import numpy as np
from matplotlib import pyplot as plt
 
cap = cv.VideoCapture(0)

while True:
    # Take each frame
    _, frame = cap.read()

    gray = cv.cvtColor(frame, cv.COLOR_BGR2GRAY)

    assert gray is not None, "file could not be read, check with os.path.exists()"
    ret,thresh1 = cv.threshold(gray,127,255,cv.THRESH_BINARY)
     
    cv.imshow('frame', thresh1)

    if cv.waitKey(1) == ord('q'):
        break
