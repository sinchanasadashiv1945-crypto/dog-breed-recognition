import os
import tensorflow as tf
from keras.utils import image_dataset_from_directory
from keras.applications import MobileNetV2
from keras.models import Sequential
from keras.layers import Dense, GlobalAveragePooling2D, Dropout
from keras.optimizers import Adam
import tkinter as tk
from tkinter import filedialog, Label, Button
from PIL import Image, ImageTk
import numpy as np
import cv2

# --- Dataset Path ---
dataset_path = "C:/Users/sinch/Downloads/archive/data"

# --- Load Dataset (NEW METHOD) ---
train_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="training",
    seed=123,
    image_size=(224,224),
    batch_size=32
)

val_ds = image_dataset_from_directory(
    dataset_path,
    validation_split=0.2,
    subset="validation",
    seed=123,
    image_size=(224,224),
    batch_size=32
)

class_names = train_ds.class_names

total_images = 0
for class_name in class_names:
    class_folder = os.path.join(dataset_path, class_name)
    num_images = len(os.listdir(class_folder))
    print(class_name, ":", num_images)
    total_images += num_images

print("Total images:", total_images)

# Normalize
train_ds = train_ds.map(lambda x, y: (x/255.0, y))
val_ds = val_ds.map(lambda x, y: (x/255.0, y))

# --- Model ---
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(224,224,3))

for layer in base_model.layers[:-20]:
    layer.trainable = False

model = Sequential([
    base_model,
    GlobalAveragePooling2D(),
    Dropout(0.5),
    Dense(128, activation='relu'),
    Dense(len(class_names), activation='softmax')
])

model.compile(optimizer=Adam(0.0001), loss='sparse_categorical_crossentropy', metrics=['accuracy'])

# --- Train or Load ---
if os.path.exists("dog_model.keras"):
    model = tf.keras.models.load_model("dog_model.keras")
    print("Loaded saved model")
else:
    print("Training model...")
    model.fit(train_ds, validation_data=val_ds, epochs=10)

    loss, acc = model.evaluate(val_ds)
    print(f"Accuracy: {acc*100:.2f}%")

    model.save("dog_model.keras")
    print("Model saved")

# --- Prediction ---
def predict_image(path):
    img = cv2.imread(path)
    img = cv2.resize(img, (224,224))
    img = img / 255.0
    img = np.expand_dims(img, axis=0)

    pred = model.predict(img)
    idx = np.argmax(pred)
    return class_names[idx]

# --- GUI ---
def open_file():
    file_path = filedialog.askopenfilename()
    if file_path:
        result = predict_image(file_path)

        img = Image.open(file_path).resize((200,200))
        img_tk = ImageTk.PhotoImage(img)
        panel.config(image=img_tk)
        panel.image = img_tk

        label_result.config(text="Prediction: " + result)

# --- UI ---
root = tk.Tk()
root.title("Dog Breed Classifier")

Button(root, text="Choose Image", command=open_file).pack(pady=10)

panel = Label(root)
panel.pack()

label_result = Label(root, text="", font=("Arial", 14))
label_result.pack(pady=10)

root.mainloop()