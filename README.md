# 🩺 Pneumonia Detection from Chest X-Ray Images

This project builds a deep learning model to classify chest X-ray images as **Pneumonia** or **Normal** using Convolutional Neural Networks (CNNs) with TensorFlow/Keras.

---

## 🧩 1. Importing Libraries

Essential libraries used:
- **TensorFlow/Keras** for building/training the model  
- **OpenCV, NumPy, Matplotlib** for image processing and visualization  
- **Scikit-learn** for data splitting and evaluation  

---

## 🗂️ 2. Data Loading and Preprocessing

- Function `get_training_data()`:
  - Loads `.jpeg` images from `PNEUMONIA` and `NORMAL` folders
  - Resizes them to **150×150**
  - Labels: `0 = NORMAL`, `1 = PNEUMONIA`

---

## 📦 3. Image Normalization

All images are scaled by dividing pixel values by **255** to normalize input between **0 and 1**.

---

## 🧪 4. Train-Test Split

Uses `train_test_split()` to divide the dataset into **training** and **validation** sets.

---

## 🧠 5. CNN Model Architecture

Model built using Keras with layers:
- `Conv2D` for feature extraction  
- `MaxPooling2D` to downsample  
- `Dropout` for regularization  
- `BatchNormalization` for faster convergence  
- `Flatten` + `Dense` for final classification  

---

## 🛑 6. Callbacks

Includes:
- `EarlyStopping`: Halts training on stagnation  
- `ModelCheckpoint`: Saves best model  
- `ReduceLROnPlateau`: Decreases LR on accuracy plateau  

---

## 🏋️ 7. Training the Model

Trained using `.fit()` with validation data monitoring and active callbacks.

---

## 📊 8. Evaluation

- Predictions made on test set  
- Evaluated with **Confusion Matrix** & **Classification Report** (Accuracy, Precision, Recall)

---

> 🧠 A practical application of CNNs for real-world medical diagnosis through deep learning.
