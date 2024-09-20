# from ultralytics import YOLO

# # Load YOLOv10n model from scratch
# model = YOLO("yolov10n")

# # Train the model
# model.train(data=, epochs = 100, imgsz)

from ultralytics import YOLO
import cv2 as cv
import numpy as np
import torch

# Load a pre-trained YOLOv10n model
model = YOLO("yolov10n.pt")

#cap = cv.VideoCapture('testTubeVid.MOV')

# fourcc = cv.VideoWriter_fourcc(*'DIVX')
# out = cv.VideoWriter('output1.avi', cv.VideoWriter_fourcc(*'MJPG'), 20.0, (1280, 720))

while True:
    # Take each frame
    _, frame = cap.read()

    # Perform object detection on an image
    results = model(frame)

    # Display the results
    print(results[0].type)

    # out.write(results[0].cpu().numpy())
    cv.imshow('frame', results[0].cpu().numpy())

    if cv.waitKey(1) == ord('q'):
        break

cap.release()
# out.release()
cv.destroyAllWindows()