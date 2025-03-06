import os
import face_recognition
import pickle

dataset_dir = "dataset/"
encodings = {}
for person in os.listdir(dataset_dir):
    encodings[person] = []
    for image in os.listdir(f"{dataset_dir}/{person}"):
        img_path = f"{dataset_dir}/{person}/{image}"
        image = face_recognition.load_image_file(img_path)
        encoding = face_recognition.face_encodings(image)
        if encoding:
            encodings[person].append(encoding[0])

with open("models/face_encodings.pkl", "wb") as f:
    pickle.dump(encodings, f)

print("Training completed. Encodings saved!")
