# ===============================
# Mask Detection Real-Time Script
# ===============================
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array
from keras.models import load_model
from imutils.video import VideoStream
import numpy as np
import imutils
import cv2
import time

# -------------------------------
# Function to detect faces and predict masks
# -------------------------------
def detect_and_predict_mask(frame, faceNet, maskNet):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (224,224), (104.0,177.0,123.0))
    faceNet.setInput(blob)
    detections = faceNet.forward()

    faces = []
    locs = []
    preds = []

    for i in range(detections.shape[2]):
        confidence = detections[0,0,i,2]
        if confidence > 0.5:
            box = detections[0,0,i,3:7] * np.array([w,h,w,h])
            (startX, startY, endX, endY) = box.astype("int")
            (startX, startY) = max(0, startX), max(0, startY)
            (endX, endY) = min(w-1, endX), min(h-1, endY)

            if endX <= startX or endY <= startY:
                continue  # skip invalid boxes

            face = frame[startY:endY, startX:endX]
            face = cv2.cvtColor(face, cv2.COLOR_BGR2RGB)
            face = cv2.resize(face, (224,224))
            face = img_to_array(face)
            # Do NOT preprocess if your model has preprocessing inside
            # face = preprocess_input(face)

            faces.append(face)
            locs.append((startX, startY, endX, endY))

    if len(faces) > 0:
        faces = np.array(faces, dtype="float32")
        preds = maskNet.predict(faces, batch_size=32)

    return (locs, preds)


# -------------------------------
# Load face detector model
# -------------------------------
prototxtPath = r"face_detector\deploy.prototxt"
weightsPath = r"face_detector\res10_300x300_ssd_iter_140000.caffemodel"
faceNet = cv2.dnn.readNet(prototxtPath, weightsPath)

# Load your trained mask detection model
maskNet = load_model("mask_detector.keras")

# -------------------------------
# Start video stream
# -------------------------------
print("[INFO] Starting video stream...")
vs = VideoStream(src=0).start()
time.sleep(2.0)

# -------------------------------
# Real-time detection loop
# -------------------------------
while True:
    frame = vs.read()
    frame = imutils.resize(frame, width=400)

    # Detect faces and predict masks
    (locations, preds) = detect_and_predict_mask(frame, faceNet, maskNet)

    if locations is not None and preds is not None:
        for (box, pred) in zip(locations, preds):
            (startX, startY, endX, endY) = box
            (with_mask, without_mask) = pred

            # Debug: print prediction probabilities
            print(f"Mask: {with_mask:.2f}, NoMask: {without_mask:.2f}")

            label = "Mask" if with_mask > without_mask else "No Mask"
            color = (0, 255, 0) if label == "Mask" else (0, 0, 255)
            label_text = f"{label}: {max(with_mask, without_mask) * 100:.2f}%"

            cv2.putText(frame, label_text, (startX, startY - 10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.45, color, 2)
            cv2.rectangle(frame, (startX, startY), (endX, endY), color, 2)

    cv2.imshow("Mask Detector", frame)
    key = cv2.waitKey(1) & 0xFF

    if key == ord("q"):
        break

# -------------------------------
# Cleanup
# -------------------------------
cv2.destroyAllWindows()
vs.stop()