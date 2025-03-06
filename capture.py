import cv2
import os

# Initialize the webcam
cap = cv2.VideoCapture(0)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

name = input("Enter your name: ").strip()
folder = f"dataset/{name}"
os.makedirs(folder, exist_ok=True)

count = 0
while True:
    ret, frame = cap.read()
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.3, 5)
    
    for (x, y, w, h) in faces:
        count += 1
        face = frame[y:y+h, x:x+w]
        cv2.imwrite(f"{folder}/{count}.jpg", face)  # Save the image
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
    
    cv2.imshow("Capturing Faces", frame)
    if cv2.waitKey(1) & 0xFF == ord('q') or count >= 30:
        break

cap.release()
cv2.destroyAllWindows()
print(f"Collected {count} images for {name}")
