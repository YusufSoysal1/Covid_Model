import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import os

# Veri setinin yolu
dataset_path = "./COVID-19_Radiography_Dataset/"

# Veri artırma (augmentation) işlemi
train_datagen = ImageDataGenerator(
    rescale=1./255,           # Piksel değerlerini 0-1 aralığına getir
    rotation_range=20,        # Döndürme
    zoom_range=0.15,          # Zoom
    width_shift_range=0.2,    # Yatay kaydırma
    height_shift_range=0.2,   # Dikey kaydırma
    shear_range=0.15,         # Kaydırma (shear)
    horizontal_flip=True,     # Yatay çevirme
    validation_split=0.2      # %20 doğrulama verisi
)

# Eğitim verisi yükleyici
train_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),   # Resim boyutları
    batch_size=32,
    class_mode='categorical',
    subset='training'
)

# Doğrulama verisi yükleyici
val_generator = train_datagen.flow_from_directory(
    dataset_path,
    target_size=(224, 224),
    batch_size=32,
    class_mode='categorical',
    subset='validation'
)

# Model kuruyoruz
model = Sequential([
    Conv2D(32, (3, 3), activation='relu', input_shape=(224, 224, 3)),
    MaxPooling2D(2, 2),

    Conv2D(64, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Conv2D(128, (3, 3), activation='relu'),
    MaxPooling2D(2, 2),

    Flatten(),

    Dense(128, activation='relu'),
    Dropout(0.5),

    Dense(4, activation='softmax')
])

# Modeli derliyoruz
model.compile(
    optimizer='adam',
    loss='categorical_crossentropy',
    metrics=['accuracy']
)

# Modelin özetini yazdır (kısaca yapısı gözükür)
model.summary()

# Modeli eğitiyoruz
history = model.fit(
    train_generator,
    epochs=10,
    validation_data=val_generator
)

# Modeli kaydet (istersen ileride kullanırsın)
model.save("covid_pneumonia_classifier.h5")

# Modelle tahmin yapma örneği
import numpy as np
from tensorflow.keras.preprocessing import image

# Test için bir resim yükleyelim
test_img_path = dataset_path + "COVID/1COVID-19.png"  # örnek bir resim yolu
img = image.load_img(test_img_path, target_size=(224, 224))
img_array = image.img_to_array(img)
img_array = np.expand_dims(img_array, axis=0)
img_array /= 255.0

# Tahmin yap
prediction = model.predict(img_array)
predicted_class = np.argmax(prediction)

# Etiketleri yazalım
labels = list(train_generator.class_indices.keys())
print(f"Bu resim büyük ihtimalle: {labels[predicted_class]}")