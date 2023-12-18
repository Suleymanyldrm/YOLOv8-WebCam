from ultralytics import YOLO
import cv2
import cvzone
import math
import time

prev_frame_time = 0
new_frame_time = 0

cap = cv2.VideoCapture(0)

model = YOLO("shoes.pt")

while True:
    new_frame_time = time.time()
    success, img = cap.read()
    results = model(img, stream=True)
    for r in results:
        boxes = r.boxes
        for box in boxes:
            # Bounding Box
            x1, y1, x2, y2 = box.xyxy[0]
            x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
            cv2.rectangle(img,(x1,y1),(x2,y2),(255,0,255),3)
            w, h = x2 - x1, y2 - y1
            cvzone.cornerRect(img, (x1, y1, w, h))

            # bulanıklastirma
            imgCrop = img[y1:y1 + h, x1:x1 + w]
            imgBlur = cv2.blur(imgCrop, (35, 35))
            img[y1:y1 + h, x1:x1 + w] = imgBlur

            # Confidence
            conf = math.ceil((box.conf[0] * 100)) / 100
            # Class Name
            cls = int(box.cls[0])

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("fps: ", fps)

    cv2.imshow("Image", img)
    cv2.waitKey(1)
