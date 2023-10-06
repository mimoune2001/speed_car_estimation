#importing librairies
import cv2 as cv
from ultralytics import YOLO
from functions import create_dataframe_from_boxes
from tracker import Tracker
import time


#line coordinate
cy1 = 322
cy2 = 368
offset = 5
line1=[(274,cy1),(814,cy1)]
line2=[(177,cy2),(927,cy2)]

cap = cv.VideoCapture("video.mp4")
model = YOLO('yolov8s.pt')
tracker = Tracker()
cars_up ={}
cars_down = {}


while True:
    ret,frame = cap.read()
    frame = cv.resize(frame,(1020,500))
    cv.line(frame,line1[0],line1[1],(255,255,255),1)
    cv.putText(frame, "L1", line1[0], cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv.line(frame,line2[0],line2[1],(255,255,255),1)
    cv.putText(frame, "L2", line2[0], cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    results = model.predict(frame)
    df = create_dataframe_from_boxes(results[0].boxes)
    cars_pos =[]
    for index,row in df.iterrows():
        if row['Class'] == 2:
            cars_pos.append([row['X1'],row['Y1'],row['X2'],row['Y2']])

    cars_pos_id = tracker.update(cars_pos)
    for box in cars_pos_id:
        x3, y3, x4, y4, id = box
        cx = int(x3 + x4) // 2
        cy = int(y3 + y4) // 2
        cv.circle(frame, (cx, cy), 4, (0, 0, 255), -1)
        # cv.putText(frame, str(id), (cx, cy), cv.FONT_HERSHEY_COMPLEX, 0.8, (0, 255, 255), 2)
        if cy >= (cy1-offset) and cy <= (cy1+offset):
            if id not in cars_up:
                cars_down[id] = time.time()
            else:
                duration = time.time() - cars_up[id]
                speed = (10//duration)*3.6
                cv.putText(frame, str(speed)+"Km/h", (cx, cy), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
        if cy >= (cy2 -offset) and cy <= (cy2+offset):
            if id not in cars_down:
                cars_up[id] = time.time()
            else:
                duration = time.time() - cars_down[id]
                speed = (10 // duration) * 3.6
                cv.putText(frame, str(speed)+"Km/h", (cx, cy), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv.putText(frame,"Cars Up :"+str(len(cars_up.keys())), (10, 30), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv.putText(frame,"Cars down :"+str(len(cars_down.keys())), (10,50), cv.FONT_HERSHEY_COMPLEX, 0.8, (255, 255, 255), 2)
    cv.imshow("Speed car estimator", frame)
    if cv.waitKey(1) & 0xFF == 27:
        break
cv.destroyAllWindows()
cap.release()


























