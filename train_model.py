import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
from tensorflow.keras.applications import VGG16
import json  # Add this to save class indices

# Directories for training, validation, and testing
train_dir = 'PlantVillage/train'
validation_dir = 'PlantVillage/validation'
test_dir = 'PlantVillage/test'

# Parameters
img_height, img_width = 256, 256
batch_size = 32
epochs = 20

# Data Augmentation and Preprocessing
train_datagen = ImageDataGenerator(
    rescale=1./255,  # Normalize pixel values to [0, 1]
    rotation_range=40,
    width_shift_range=0.2,
    height_shift_range=0.2,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input  # VGG16 specific preprocessing
)

validation_datagen = ImageDataGenerator(
    rescale=1./255,  # Only rescale for validation and test data
    preprocessing_function=tf.keras.applications.vgg16.preprocess_input  # VGG16 specific preprocessing
)

# Load images from directories with augmentation
train_generator = train_datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

validation_generator = validation_datagen.flow_from_directory(
    validation_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# VGG16 Model Architecture (Base Model)
base_model = VGG16(weights='/Users/shweta/Crop Disease Recognition/model/vgg16_weights_tf_dim_ordering_tf_kernels_notop.h5', include_top=False, input_shape=(img_height, img_width, 3))

# Freeze the layers of the VGG16 base model
base_model.trainable = False

# CNN Model Architecture using VGG16 as a base
model = models.Sequential([
    base_model,  # Add VGG16 as the base
    layers.GlobalAveragePooling2D(),  # Pooling layer
    layers.Dense(512, activation='relu'),
    layers.Dense(len(train_generator.class_indices), activation='softmax')  # Output layer
])

# Compile the model with categorical crossentropy for multi-class classification
model.compile(loss='categorical_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

# Train the model
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size
)

# Save the class indices to a JSON file
with open('class_indices.json', 'w') as f:
    json.dump(train_generator.class_indices, f)  # Save class labels and indices

# Save the model after training
model.save('crop_disease_vgg16_model.h5')

# Evaluate the model on the test dataset
test_datagen = ImageDataGenerator(rescale=1./255, preprocessing_function=tf.keras.applications.vgg16.preprocess_input)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    batch_size=batch_size,
    class_mode='categorical'
)

# Test the model
test_loss, test_acc = model.evaluate(test_generator)
print(f"Test Accuracy: {test_acc}")
