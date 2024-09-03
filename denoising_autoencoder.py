import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras import layers, models
import matplotlib.pyplot as plt

# Define image dimensions and directory paths
img_height, img_width = 224, 224  # Adjust according to your dataset
train_dir = 'C:/Users/Lenovo/PycharmProjects/Denoising Autoencoder for Medical Image Enhancement/COVID-19_Radiography_Dataset'
test_dir = 'C:/Users/Lenovo/PycharmProjects/Denoising Autoencoder for Medical Image Enhancement/COVID-19_Radiography_Dataset'

# Load and preprocess the data using ImageDataGenerator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_height, img_width),
    color_mode="grayscale",
    batch_size=32,
    class_mode="input",
    subset='training'
)

validation_generator = datagen.flow_from_directory(
    test_dir,
    target_size=(img_height, img_width),
    color_mode="grayscale",
    batch_size=32,
    class_mode="input",
    subset='validation'
)

# Build the autoencoder model
input_img = layers.Input(shape=(img_height, img_width, 1))

# Encoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(input_img)
x = layers.MaxPooling2D((2, 2), padding='same')(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
encoded = layers.MaxPooling2D((2, 2), padding='same')(x)

# Decoder
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(encoded)
x = layers.UpSampling2D((2, 2))(x)
x = layers.Conv2D(32, (3, 3), activation='relu', padding='same')(x)
x = layers.UpSampling2D((2, 2))(x)
decoded = layers.Conv2D(1, (3, 3), activation='sigmoid', padding='same')(x)

autoencoder = models.Model(input_img, decoded)
autoencoder.compile(optimizer='adam', loss='binary_crossentropy')

# Train the model
autoencoder.fit(
    train_generator,
    epochs=50,
    validation_data=validation_generator
)

# Predict and visualize the results
test_images, _ = next(validation_generator)
decoded_imgs = autoencoder.predict(test_images)

# Display the original, noisy, and denoised images
n = 10
plt.figure(figsize=(20, 4))
for i in range(n):
    # Display original
    ax = plt.subplot(3, n, i + 1)
    plt.imshow(test_images[i].reshape(img_height, img_width), cmap='gray')
    plt.axis('off')

    # Display denoised
    ax = plt.subplot(3, n, i + 1 + n)
    plt.imshow(decoded_imgs[i].reshape(img_height, img_width), cmap='gray')
    plt.axis('off')
plt.show()
