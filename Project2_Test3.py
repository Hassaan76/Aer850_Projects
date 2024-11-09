# Imports
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import keras_tuner as kt
import tensorflow as tf
from keras import models, layers
from tensorflow.python.client import device_lib
import datetime

#STEP 1:
# Define image shape and batch size
IMG_WIDTH, IMG_HEIGHT = 500, 500
BATCH_SIZE = 32

# Paths to data directories
train_dir = r"C:\Users\hassa\Documents\GitHub\Projects\Project2\train"
valid_dir = r"C:\Users\hassa\Documents\GitHub\Projects\Project2\valid"
test_dir = r"C:\Users\hassa\Documents\GitHub\Projects\Project2\test"

# Data Augmentation and Rescaling for Training Data
train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

# Rescaling for Validation Data
valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)

# Data Generators for Training and Validation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

validation_generator = valid_datagen.flow_from_directory(
    valid_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# Data Generator for Testing Data (optional, only rescaling)
test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)


# Constants (These should be adapted based on your dataset)
BATCH_SIZE = 32  # Example, adjust based on your system's memory
IMG_HEIGHT = 500
IMG_WIDTH = 500
NUM_CLASSES = 3  # Assuming you have 3 classes to predict

#STEP 2: Define initial CNN model
model = models.Sequential()

# Convolutional Layer 1
model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 2
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 3
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 4
model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Convolutional Layer 5
model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
model.add(layers.Flatten())

# Fully Connected Dense Layer
model.add(layers.Dense(16, activation='relu'))
model.add(layers.Dropout(0.5))  # Dropout layer to reduce overfitting

# Output layer (3 neurons for 3 classes)
model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

# Compile the model with default Adam optimizer
model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

# Model summary
model.summary()

# STEP 3: Tune hyperparameters (alternative model with hyperparameter adjustments)
tuned_model = models.Sequential()

# Convolutional Layer with LeakyReLU activation function
tuned_model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
tuned_model.add(layers.LeakyReLU(alpha=0.1))
tuned_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Second Convolutional Layer with more filters and ReLU activation
tuned_model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
tuned_model.add(layers.MaxPooling2D(pool_size=(2, 2)))

# Flatten layer
tuned_model.add(layers.Flatten())

# Dense layer with ELU activation function and Dropout
tuned_model.add(layers.Dense(16))
tuned_model.add(layers.ELU(alpha=1.0))
tuned_model.add(layers.Dropout(0.5))

# Output layer (3 neurons for 3 classes)
tuned_model.add(layers.Dense(NUM_CLASSES, activation='softmax'))

# Compile the tuned model with a smaller learning rate
tuned_model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0005), 
                    loss='categorical_crossentropy', 
                    metrics=['accuracy'])

# Model summary of tuned model
tuned_model.summary()

# Train the tuned model
history = tuned_model.fit(
    train_generator, 
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=2,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

#STEP: 4
# Plot training & validation accuracy values
plt.figure(figsize=(12, 5))

# Accuracy Plot
plt.subplot(1, 2, 1)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.title('Training and Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend(loc="lower right")

# Loss Plot
plt.subplot(1, 2, 2)
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(loc="upper right")

plt.show()
