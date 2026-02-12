import keras
from keras import layers
from keras.applications import MobileNetV2
from tensorflow.keras.layers import AveragePooling2D, Dropout, Flatten, Dense, Input
from keras.models import Model
from tensorflow.keras.optimizers import Adam
from keras.applications.mobilenet_v2 import preprocess_input
from keras.preprocessing.image import img_to_array, load_img
from keras.utils import to_categorical
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report
import matplotlib.pyplot as plt
import numpy as np
import os

INIT_LR = 1e-4
EPOCHS = 20
BS = 32

Directory = r"Mask Dataset/facemask-dataset/dataset"
Categories = ["with_mask","without_mask"]
print("[INFO] loading images...")
data = []
labels = []
for cat in Categories:
    path = os.path.join(Directory, cat)
    for img in os.listdir(path):
        img_path = os.path.join(path,img)
        image = load_img(img_path, target_size=(224,224), color_mode="rgb")
        image = img_to_array(image)
        image = preprocess_input(image)
        data.append(image)
        labels.append(cat)

lb = LabelBinarizer()
labels = lb.fit_transform(labels)
labels = to_categorical(labels)
data = np.array(data,dtype="float32")
labels = np.array(labels)
(trainX, testX, trainY, testY) = train_test_split(data,labels,test_size=0.2, stratify=labels, random_state=42)
aug = keras.Sequential([
    layers.RandomFlip("horizontal"),
    layers.RandomRotation(0.2),
    layers.RandomZoom(0.15),
    layers.RandomTranslation(0.2, 0.2),
])
baseModel = MobileNetV2(weights="imagenet", include_top=False, input_tensor=Input(shape=(224,224,3)))
headModel = baseModel.output
headModel = AveragePooling2D(pool_size=(7,7))(headModel)
headModel = Flatten(name="flatten")(headModel)
headModel = Dense(128,activation='relu')(headModel)
headModel = Dropout(0.5)(headModel)
headModel = Dense(2,activation="softmax")(headModel)
inputs = Input(shape=(224,224,3))

x = aug(inputs)  # augmentation layer inside model
x = baseModel(x, training=False)

x = AveragePooling2D(pool_size=(7,7))(x)
x = Flatten(name="flatten")(x)
x = Dense(128, activation="relu")(x)
x = Dropout(0.5)(x)
outputs = Dense(2, activation="softmax")(x)

model = Model(inputs, outputs)
model = Model(inputs=baseModel.input, outputs=headModel)
for layer in baseModel.layers:
    layer.trainable = False
print("[INFO] training head")
opt = Adam(learning_rate=INIT_LR)
model.compile(
    loss="categorical_crossentropy",
    optimizer=opt,
    metrics=["accuracy"]
)
print("[INFO] training head...")
H = model.fit(
    aug(trainX, training=True),
    trainY,
    batch_size=BS,
    validation_data=(testX, testY),
    epochs=EPOCHS
)
print("[INFO] evaluating Networks")
pred = model.predict(testX, batch_size=BS)
pred = np.argmax(pred, axis=1)
print(classification_report(testY.argmax(axis=1),pred, target_names=lb.classes_))
print("Saving Mask Detector model.")
model.save("mask_detector.model", save_format="h5")
N = EPOCHS
plt.style.use("ggplot")
plt.figure()
plt.plot(np.arange(0, N), H.history["loss"], label="train_loss")
plt.plot(np.arange(0, N), H.history["val_loss"], label="val_loss")
plt.plot(np.arange(0, N), H.history["accuracy"], label="train_acc")
plt.plot(np.arange(0, N), H.history["val_accuracy"], label="val_acc")
plt.title("Training Loss and Accuracy")
plt.xlabel("Epoch #")
plt.ylabel("Loss/Accuracy")
plt.legend(loc="lower left")
plt.savefig("plot.png")

















