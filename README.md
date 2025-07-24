Pneumonia Detection from Chest X-Ray Images


🧩 1. Importing Libraries
The code starts by importing important libraries like:
TensorFlow/Keras: for building and training the neural network.
OpenCV, NumPy, Matplotlib: for image handling and visualization.
Scikit-learn: for splitting data and evaluating the model.


🗂️ 2. Data Loading and Preprocessing
A function named get_training_data():
Loads .jpeg images from two folders: PNEUMONIA and NORMAL.
Resizes each image to 150x150.
Labels each image (0 for NORMAL, 1 for PNEUMONIA).


📦 3. Image Normalization
Images are normalized by dividing pixel values by 255 to scale them between 0 and 1 — a standard practice for faster and stable training.


🧪 4. Train-Test Split
Data is split into training and validation sets using train_test_split() to evaluate the model’s generalization.


🧠 5. CNN Model Architecture
The model is built using Keras:
Conv2D layers to extract image features.
MaxPooling2D layers to reduce spatial dimensions.
Dropout to prevent overfitting.
BatchNormalization to stabilize and speed up training.
Flatten to convert 2D data to 1D.
Dense layers for classification.


🛑 6. Callbacks
EarlyStopping: Stops training if the model stops improving.
ModelCheckpoint: Saves the best-performing model during training.
ReduceLROnPlateau: Lowers the learning rate when accuracy plateaus.


🏋️ 7. Training the Model
The model is trained using .fit() with training and validation data.


📊 8. Evaluation
After training:
Predictions are made on test data.
A confusion matrix and classification report show metrics like accuracy, precision, and recall.
