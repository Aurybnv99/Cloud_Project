import tensorflow as tf
import numpy as np
import pandas as pd
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.model_selection import train_test_splitfrom tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization # Import BatchNormalization# type: ignore
from tensorflow.keras.optimizers import Adam# type: ignore
from tensorflow.keras.preprocessing.image import ImageDataGenerator # Import ImageDataGenerator# type: ignore
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau # Import Callbacks# type: ignore
from tensorflow.keras import regularizers # Import regularizers# type: ignore

def create_improved_model(l2_reg=0.001): # Add L2 regularization factor
    model = Sequential()

    # Block 1
    model.add(Conv2D(32, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg), input_shape=(48, 48, 1)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Consider adding Dropout here: model.add(Dropout(0.25))

    # Block 2
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Consider adding Dropout here: model.add(Dropout(0.25))

    # Block 3
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Consider adding Dropout here: model.add(Dropout(0.25))

    # Block 4 (Optional - could remove if overfitting persists)
    model.add(Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Consider adding Dropout here: model.add(Dropout(0.25))


    # Flatten
    model.add(Flatten())

    # Fully connected layer 1
    model.add(Dense(512, kernel_regularizer=regularizers.l2(l2_reg)))
    model.add(BatchNormalization())
    model.add(tf.keras.layers.Activation('relu'))
    model.add(Dropout(0.5)) # Keep existing Dropout

    # Output layer
    model.add(Dense(7, activation='softmax'))

    optimizer = Adam(learning_rate=0.001) # Start with the original LR
    model.compile(optimizer=optimizer, loss='categorical_crossentropy', metrics=['accuracy'])
    return model



# Load the dataset
data = pd.read_csv('/content/drive/MyDrive/fer2013.csv')

# Preprocess the data
X = []
y = []
for index, row in data.iterrows():
    pixels = np.array(row['pixels'].split(), dtype='float32')
    # Ensure pixels form a 48x48 image before reshaping
    if pixels.shape[0] == 48*48:
        X.append(pixels.reshape(48, 48, 1))
        y.append(row['emotion'])
    else:
        print(f"Skipping row {index} due to incorrect number of pixels: {pixels.shape[0]}")


X = np.array(X) / 255.0  # Normalize pixel values
y = to_categorical(np.array(y), num_classes=7)

# Split the data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y) # type: ignore # Use stratify for imbalanced datasets

# --- Data Augmentation ---
datagen = ImageDataGenerator(
    rotation_range=15,       # Randomly rotate images
    width_shift_range=0.1,   # Randomly shift images horizontally
    height_shift_range=0.1,  # Randomly shift images vertically
    shear_range=0.1,         # Shear transformation
    zoom_range=0.1,          # Randomly zoom images
    horizontal_flip=True,    # Randomly flip images horizontally
    fill_mode='nearest'      # Strategy for filling newly created pixels
)

# Only fit the generator on the training data
datagen.fit(X_train)

# --- Callbacks ---
early_stopping = EarlyStopping(
    monitor='val_accuracy',  # Monitor validation accuracy
    patience=15,             # Number of epochs with no improvement after which training will be stopped
    restore_best_weights=True, # Restore model weights from the epoch with the best val_accuracy
    verbose=1
)

reduce_lr = ReduceLROnPlateau(
    monitor='val_loss',    # Monitor validation loss
    factor=0.2,            # Factor by which the learning rate will be reduced. new_lr = lr * factor
    patience=5,            # Number of epochs with no improvement after which learning rate will be reduced.
    min_lr=0.00001,        # Lower bound on the learning rate.
    verbose=1
)

# --- Create and Train Model ---
model = create_improved_model()
model.summary() # Print model summary

batch_size = 32 # Often helps generalization to use slightly larger batch sizes with augmentation/BN
epochs = 100    # Increase epochs, early stopping will handle stopping time

# Use fit with the generator
history = model.fit(
    datagen.flow(X_train, y_train, batch_size=batch_size),
    steps_per_epoch=len(X_train) // batch_size,
    epochs=epochs,
    validation_data=(X_test, y_test),
    callbacks=[early_stopping, reduce_lr] # Add callbacks
)

# Evaluate the final model (with best weights restored by EarlyStopping)
loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
print(f'Final Test Loss: {loss:.4f}')
print(f'Final Test Accuracy: {accuracy:.4f}')

# Save the improved model
model.save('emotion_detection_model_improved.h5')