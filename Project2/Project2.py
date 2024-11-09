# Hassaan Arif | 501064115

import zipfile
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras import layers, models
from keras.src.legacy.preprocessing.image import ImageDataGenerator
from keras.optimizers import Adam
from keras.preprocessing import image


zip_path = r"C:\Users\hassa\Documents\GitHub\Projects\Project2\Project 2 Data.zip\Data"
extract_to_path = r"C:/Users/hassa/Documents/GitHub/Projects/Project2"

with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    zip_ref.extractall(extract_to_path)
print("Dataset extracted successfully.")

# Set paths to directories after extraction
train_dir = os.path.join(extract_to_path, 'Project 2 Data', 'Train')
validation_dir = os.path.join(extract_to_path, 'Project 2 Data', 'Validation')
test_dir = os.path.join(extract_to_path, 'Project 2 Data', 'Test')

# Step 2: Data Processing
# Image shape
img_shape = (500, 500)

# Data augmentation for training and validation datasets
train_datagen = ImageDataGenerator(
    rescale=1./255,
    shear_range=0.2,
    zoom_range=0.2
)

validation_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=img_shape,
    batch_size=32,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=img_shape,
    batch_size=32,
    class_mode='categorical'
)

# Step 3: Neural Network Architecture Design
model = models.Sequential([
    layers.Conv2D(32, (3, 3), activation='relu', input_shape=(500, 500, 3)),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(64, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Conv2D(128, (3, 3), activation='relu'),
    layers.MaxPooling2D((2, 2)),
    layers.Flatten(),
    layers.Dense(512, activation='relu'),
    layers.Dropout(0.5),
    layers.Dense(3, activation='softmax')
])

# Step 4: Hyperparameter Analysis
model.compile(
    optimizer=Adam(learning_rate=0.001),  # Updated Adam import
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Step 5: Model Training
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // 32,
    epochs=20,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // 32
)

# Step 6: Model Evaluation (Plotting Loss and Accuracy)
plt.figure(figsize=(12, 4))
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.legend()
plt.title('Training and Validation Accuracy')

plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.legend()
plt.title('Training and Validation Loss')
plt.show()

# Step 7: Model Testing on New Images
# Function to load and predict an image
def predict_image(img_path):

    img = image.load_img(img_path, target_size=img_shape)
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    class_names = list(train_generator.class_indices.keys())
    predicted_class = class_names[np.argmax(prediction)]

    plt.imshow(img)
    plt.title(f"Predicted: {predicted_class}")
    plt.axis('off')
    plt.show()

# Testing the model with example images
predict_image(os.path.join(test_dir, 'crack', 'test_crack.jpg'))
predict_image(os.path.join(test_dir, 'missing-head', 'test_missinghead.jpg'))
predict_image(os.path.join(test_dir, 'paint-off', 'test_paintoff.jpg'))
