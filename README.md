# 🐶 Dog Breed Recognition using Deep Learning

## 📌 Overview
This project is a deep learning-based image classification system that identifies dog breeds from input images.  
It uses **transfer learning with MobileNetV2** to improve accuracy and reduce training time.

The project also includes a **Tkinter-based GUI application** for easy image upload and prediction.

---

## 🎯 Objective
To develop an AI system that can:
- Classify dog breeds from images
- Provide real-time prediction
- Demonstrate transfer learning in computer vision

---

## 🧠 Model Architecture
- Base Model: MobileNetV2 (pretrained on ImageNet)
- Global Average Pooling Layer
- Dropout Layer (0.5)
- Dense Layer (128 neurons, ReLU activation)
- Output Layer (Softmax classification)

---

## 🛠️ Tech Stack
- Python  
- TensorFlow / Keras  
- OpenCV  
- NumPy  
- Pillow  
- Tkinter (GUI)

---

## 📂 Dataset
- Dataset loaded using `image_dataset_from_directory`
- Images resized to 224 x 224 pixels
- Dataset split:
  - Training: 80%
  - Validation: 20%

Note: Dataset is not included in this repository due to size limitations.

---

## ⚙️ How It Works
1. User selects an image using GUI
2. Image is resized and normalized
3. Trained model processes the image
4. Output breed is predicted with confidence score
5. Result is displayed in the application window

---

## 🧾 Model File
- Trained model saved as: `dog_model.keras`

---

## 💻 Features
- Dog breed classification using CNN
- GUI-based image upload system
- Real-time prediction
- Transfer learning using MobileNetV2
- Easy-to-use interface

---

## 📸 Sample Output
- Input: Dog image  
- Output: Predicted breed with confidence score  
  Example: Golden Retriever – 92% confidence

---

## 🚀 Future Improvements
- Improve accuracy using larger dataset
- Deploy as web application (Flask / Django)
- Add real-time webcam prediction
- Enhance GUI design

---

## 👩‍💻 Author
Sinchana S  
BCA, JSS Science and Technology University, Mysuru  

---

## ⭐ Note
This project is developed for educational and research purposes to demonstrate deep learning and computer vision concepts.
