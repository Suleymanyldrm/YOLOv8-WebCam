# Ayakkabıyı Görüyor Şişeyi ve Yüzü bulanıklaştırıyor.
from ultralytics import YOLO
import cv2
import time
import cvzone
from cvzone.FaceDetectionModule import FaceDetector

prev_frame_time = 0
new_frame_time = 0

# yüz için fonksiyon
detector = FaceDetector(minDetectionCon=0.75)

classNames = ["Shoes", "Sise"]
model = YOLO('shoes_bottle.pt')

cap = cv2.VideoCapture(0)

# Video yakalama döngüsü
while cap.isOpened():
    new_frame_time = time.time()
    success, frame = cap.read()
    # Modelden Şiseyi ve Ayakkabıyı Yakala
    results = model(frame, stream=True)

    # Yüz İçin Yüzü Yakala
    frame, bboxs = detector.findFaces(frame, draw=True)
    # Yüz Bulanıklaştırma
    if bboxs:
        for i, bbox in enumerate(bboxs):
            x, y, w, h = bbox['bbox']
            if x < 0:
                x = 0
            if y < 0:
                y = 0

        imgCrop = frame[y:y + h, x:x + w]
        imgBlur = cv2.blur(imgCrop, (60, 60))
        frame[y:y + h, x:x + w] = imgBlur
    # Yüz Bulanıklaştırma

    for r in results:
        boxes = r.boxes
        for box in boxes:

            cls = int(box.cls[0])
            conf = box.conf[0]
            x1, y1, x2, y2 = map(int, box.xyxy[0])
            # Ayakkabı için kutucuk ve isim
            if classNames[cls] == "Shoes":
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                Name_Conf = f"Shoes {conf:.2f}"  # Bulma oranını yuvarlaklaştırarak yanına da sınıf ismini ekleyerek yazdırıyor
                cv2.putText(frame, Name_Conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 1)

            # Şişe Bulanıklaştırma, Sadece "Sise" Sınıfı İçin İşlem Yap
            elif classNames[cls] == "Sise":
                # Bounding Box
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 0, 255), 3)
                w, h = x2 - x1, y2 - y1
                cvzone.cornerRect(frame, (x1, y1, w, h))
                Name_Conf = f"Sise {conf:.2f}"  # Bulma oranını yuvarlaklaştırarak yazdır
                cv2.putText(frame, Name_Conf, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 0, 255), 1)

                # Blur Kodu
                imgCrop = frame[y1:y1 + h, x1:x1 + w]
                imgBlur = cv2.blur(imgCrop, (100, 100))
                frame[y1:y1 + h, x1:x1 + w] = imgBlur
            # Şişe Bulanıklaştırma

    fps = 1 / (new_frame_time - prev_frame_time)
    prev_frame_time = new_frame_time
    print("fps: ", fps)

    # Görüntünün yeni boyutlarını belirle
    width = 800  # yeni genişlik
    height = 608  # yeni yükseklik
    dim = (width, height)
    resized_frame = cv2.resize(frame, dim, interpolation=cv2.INTER_AREA)
    cv2.imshow("Image", resized_frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
    # Görüntünün yeni boyutlarını belirle

cap.release()
cv2.destroyAllWindows()
