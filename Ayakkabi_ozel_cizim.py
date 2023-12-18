from ultralytics import YOLO
import cv2
import cvzone
import math
import time

prev_frame_time = 0
new_frame_time = 0

model = YOLO('shoes.pt')

cap = cv2.VideoCapture(0)

# Video yakalama döngüsü
while cap.isOpened():
    new_frame_time = time.time()
    success, frame = cap.read()
    results = model(frame, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(frame, (x1, y1, w, h))

            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

    fps = 1/(new_frame_time-prev_frame_time)
    prev_frame_time = new_frame_time
    print("fps: ", fps)
    print("Are there any pictures? :", success)

    cv2.imshow('Akış', frame)
    cv2.waitKey(1)
