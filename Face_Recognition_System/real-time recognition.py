import cv2 as cv
import face_recognition
import numpy as np
import joblib
clf = joblib.load('face_recognizer_svm.pkl')
cap = cv.VideoCapture(0)
if not cap.isOpened():
    print("Error: Cannot open webcam")
    exit()
while True:
    ret, frame = cap.read()
    # If frame is not read correctly, break
    if not ret:
        print("Error: Can't receive frame (stream end?). Exiting ...")
        break
    rgb = cv.cvtColor(frame,cv.COLOR_BGR2RGB)
    #small_frame = cv.resize(frame, (0,0),fx = 0.5, fy = 0.5)
    #rgb_small = cv.cvtColor(small_frame,cv.COLOR_BGR2GRAY)
    face_location = face_recognition.face_locations(rgb)
    face_encodings = face_recognition.face_encodings(rgb,face_location)
    for (top, right, bottom, left), face_encoding in zip(face_location,face_encodings):
        person_name = clf.predict([face_encoding])[0]
        probs = np.max(clf.predict_proba([face_encoding]))
        cv.rectangle(frame, (left,top),(right,bottom),(0,255,0),2 )
        cv.putText(frame, f"{person_name} ({probs:.2f})", (left, top - 10),
                    cv.FONT_HERSHEY_SIMPLEX, 0.8, (36, 255, 12), 2)




    cv.imshow('Face Recognition (Linear SVM)', frame)
    if cv.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv.destroyAllWindows()