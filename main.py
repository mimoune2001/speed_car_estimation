#importing librairies
import cv2 as cv
from ultralytics import YOLO
from functions import create_dataframe_from_boxes

cap = cv.VideoCapture("video.mp4")
model = YOLO('yolov8s.pt')

ret,frame = cap.read()
frame = cv.resize(frame,(1020,500))
results = model.predict(frame)
df = create_dataframe_from_boxes(results[0].boxes)





















