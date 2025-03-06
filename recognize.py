import cv2
import face_recognition
import pickle

# Load trained face encodings
with open("models/face_encodings.pkl", "rb") as f:
    known_encodings = pickle.load(f)

cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(rgb_frame)
    face_encodings = face_recognition.face_encodings(rgb_frame, face_locations)

    for face_encoding, (top, right, bottom, left) in zip(face_encodings, face_locations):
        name = "Unknown"
        for person, encodings in known_encodings.items():
            matches = face_recognition.compare_faces(encodings, face_encoding, tolerance=0.5)
            if True in matches:
                name = person
                break
        
        # Display the name
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(frame, name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("Face Recognition", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
