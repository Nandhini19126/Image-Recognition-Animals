import tensorflow as tf
from tensorflow.keras import datasets, layers, models

# Load CIFAR-10 Dataset
(x_train, y_train), (x_test, y_test) = datasets.cifar10.load_data()

# Normalize the Images
x_train = x_train / 255.0
x_test = x_test / 255.0

# CNN Model
model = models.Sequential([
    layers.Conv2D(32, (3,3), activation='relu',
                  input_shape=(32,32,3)),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),
    layers.MaxPooling2D((2,2)),

    layers.Conv2D(64, (3,3), activation='relu'),

    layers.Flatten(),
    layers.Dense(64, activation='relu'),
    layers.Dense(10, activation='softmax')
])

# Compiling Model
model.compile(
    optimizer='adam',
    loss='sparse_categorical_crossentropy',
    metrics=['accuracy']
)

# Training the Model
model.fit(x_train, y_train, epochs=10,
          validation_data=(x_test, y_test))

# Save the Model
model.save("image_classifier.h5")
print("Model Saved Successfully!")
