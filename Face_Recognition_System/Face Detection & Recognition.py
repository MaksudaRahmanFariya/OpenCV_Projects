import os
import face_recognition
import cv2 as cv
import numpy as np
from sklearn.svm import SVC
import joblib

# Path to your dataset
dataset_path = r"Faces_rec\lfw-funneled\lfw_funneled"

known_encodings = []
known_names = []

# Loop through each person’s folder
for person_name in os.listdir(dataset_path):
    person_dir = os.path.join(dataset_path, person_name)
    if not os.path.isdir(person_dir):
        continue  # Skip files that aren’t directories

    print(f"Processing {person_name}...")

    # Loop through each image for that person
    for file_name in os.listdir(person_dir):
        img_path = os.path.join(person_dir, file_name)

        # Skip non-image files
        if not file_name.lower().endswith(('.jpg', '.jpeg', '.png')):
            continue

        try:
            # Load image and compute encodings
            image = face_recognition.load_image_file(img_path)
            encodings = face_recognition.face_encodings(image)

            if len(encodings) > 0:
                known_encodings.append(encodings[0])
                known_names.append(person_name)

        except PermissionError:
            print(f"Skipping unreadable file: {img_path}")
            continue
        except Exception as e:
            print(f"Error processing {img_path}: {e}")
            continue

print(f"Total faces encoded: {len(known_encodings)}")

# Train Linear SVM
if len(known_encodings) > 0:
    clf = SVC(kernel='linear', probability=True)
    clf.fit(known_encodings, known_names)
    joblib.dump(clf, 'face_recognizer_svm.pkl')
    print("🎉 Model trained and saved as face_recognizer_svm.pkl")
else:
    print(" No face encodings found. Please check dataset path or image files.")
