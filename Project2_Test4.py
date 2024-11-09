import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf
from keras import models, layers
from tensorflow.python.client import device_lib
import datetime

# STEP 1: DATA PROCESSING
print(device_lib.list_local_devices())

IMG_WIDTH, IMG_HEIGHT = 500, 500
BATCH_SIZE = 32
NUM_CLASSES = 3  

train_dir = r"C:\Users\hassa\Documents\GitHub\Projects\Project2\train"
valid_dir = r"C:\Users\hassa\Documents\GitHub\Projects\Project2\valid"
test_dir = r"C:\Users\hassa\Documents\GitHub\Projects\Project2\test"

train_datagen = tf.keras.preprocessing.image.ImageDataGenerator(
    rescale=1.0/255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True
)

valid_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)


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

test_datagen = tf.keras.preprocessing.image.ImageDataGenerator(rescale=1.0/255)
test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(IMG_WIDTH, IMG_HEIGHT),
    batch_size=BATCH_SIZE,
    class_mode='categorical'
)

# STEP 2: NEURAL NETWORK ARCHITECTURE DESIGN
model = models.Sequential()


model.add(layers.Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Conv2D(filters=64, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Conv2D(filters=128, kernel_size=(3, 3), activation='relu'))
model.add(layers.MaxPooling2D(pool_size=(2, 2)))


model.add(layers.Flatten())


model.add(layers.Dense(128, activation='relu'))  
model.add(layers.Dropout(0.5))  


model.add(layers.Dense(128, activation='relu'))  
model.add(layers.Dropout(0.5)) 


model.add(layers.Dense(NUM_CLASSES, activation='softmax'))


model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])


model.summary()

# STEP 3: HYPERPARAMETER ANALYSIS
history = model.fit(
    train_generator, 
    steps_per_epoch=train_generator.samples // BATCH_SIZE,
    epochs=20, 
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // BATCH_SIZE
)

# STEP 4: MODEL EVALUATION
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


plt.tight_layout()
plt.show()

# Use - tf.keras.models.load_model('my_model.h5') - to call trained model
model.save('my_model.h5')


# Optional: Evaluate the model on the test data
# test_loss, test_acc = model.evaluate(test_generator, steps=test_generator.samples // BATCH_SIZE)
# print(f'Test accuracy: {test_acc}')
