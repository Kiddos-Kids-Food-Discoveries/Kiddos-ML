
# Import
import os
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from tensorflow.keras.models import Model
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, GlobalAveragePooling2D, Dense, Dropout, BatchNormalization
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

#define awal
IMG_SIZE = 128  
BATCH_SIZE = 32
train_dir = 'dataset/train'
val_dir = 'dataset/validation'
test_dir = 'dataset/test'
#ImageDataGenerator
train_datagen = ImageDataGenerator(
    rescale=1.0/255,
    rotation_range=20,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)
val_test_datagen = ImageDataGenerator(rescale=1.0/255)  # Normalisasi gambar tanpa augmentasi
train_data = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical' #one-hot encoding 
)
val_data = val_test_datagen.flow_from_directory(
    val_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)
test_data = val_test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_SIZE, IMG_SIZE),
    batch_size=BATCH_SIZE,
    class_mode='categorical',
    shuffle=False
)


# Load best model dan simpan selama training
model = load_model("best_model.keras")
print("Model berhasil dimuat!")

# Mendapatkan prediksi untuk data testing
y_pred = model.predict(test_data)
y_pred_classes = np.argmax(y_pred, axis=1)  # Prediksi kelas
y_true = test_data.classes  # Label sebenarnya dari data testing


def predict_image(image_path, model, class_indices):
    # Load dan preprocess gambar
    img = load_img(image_path, target_size=(IMG_SIZE, IMG_SIZE))  # resize ke 128x128
    img_array = img_to_array(img) / 255.0  # Normalisasi
    img_array = np.expand_dims(img_array, axis=0)  # Tambahkan batch dimension

    # Prediksi
    predictions = model.predict(img_array)
    predicted_class = np.argmax(predictions, axis=1)[0]  # Kelas dengan probabilitas tertinggi
    confidence = predictions[0][predicted_class] * 100

    # Mapping indeks ke nama kelas
    class_labels = {v: k for k, v in class_indices.items()}
    predicted_label = class_labels[predicted_class]

    return predicted_label, confidence


#PREDICT
# Path gambar baru
new_image_path = "test_images/23.jpg"

# Prediksi gambar baru
predicted_label, confidence = predict_image(new_image_path, model, train_data.class_indices)
print(f"Gambar diprediksi sebagai: {predicted_label} , confidence: {confidence:.2f}%")

