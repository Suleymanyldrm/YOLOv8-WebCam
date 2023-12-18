from ultralytics import YOLO
# from PIL import Image
import cv2
import cvzone
import math

classNames = ["Shoes", "Sise"]

model = YOLO('shoes_bottle.pt')

sonuc = model.predict(source="0", show=True)
# sonuc = model.predict(source="Images/cocuk_kopek.mp4", show=True)
# print(bool(sonuc))

# im1 = Image.open("Images/karisik.jpg")
# sonuc = model.predict(source=im1, show=True)
# cv2.waitKey(0)
# cv2.destroyAllWindows()

# Video dosyasını aç
cap = cv2.VideoCapture(sonuc)

# Video yakalama döngüsü
while cap.isOpened():
    ret, frame = cap.read()

    # Eğer frame okunamazsa döngüyü kır
    if not ret:
        break

    # Frame'i göster
    cv2.imshow('Video', frame)

    # 'q' tuşuna basıldığında döngüden çık
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Her şey bittiğinde kaynakları serbest bırak
cap.release()
cv2.destroyAllWindows()
