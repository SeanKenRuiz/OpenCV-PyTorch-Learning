import cv2 as cv
import numpy as np

cap = cv.VideoCapture('testTubeVid.MOV')

fourcc = cv.VideoWriter_fourcc(*'DIVX')
out = cv.VideoWriter('output1.avi', cv.VideoWriter_fourcc(*'MJPG'), 20.0, (1280, 720))

while True:
    # Take each frame
    _, frame = cap.read()

    resize_frame = cv.resize(frame, (400, 800))
    gray = cv.cvtColor(resize_frame, cv.COLOR_BGR2GRAY)

    sift = cv.SIFT_create()
    kp = sift.detect(gray,None)

    img = cv.drawKeypoints(gray,kp,resize_frame, (255, 0, 0), 
                                 flags=cv.DRAW_MATCHES_FLAGS_DRAW_RICH_KEYPOINTS)

    spl_image = np.hstack((resize_frame, img))

    out.write(spl_image)
    cv.imshow('frame', img)
    
    if cv.waitKey(1) == ord('q'):
        break

cap.release()
out.release()
cv.destroyAllWindows()