#CUDA_VISIBLE_DEVICES=3 python train.py
#watch -n 1 nvidia-smi

import os
import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
import shutil
import matplotlib
matplotlib.use('Agg') # <--- CRITICAL: Tells python "don't try to open a window"
import matplotlib.pyplot as plt

# --- CONFIGURATION ---
# We will point this to the folder you uploaded
BASE_DATA_DIR = os.path.expanduser("~/brain_project_data") 
WORKING_DIR = "tumorous_and_nontumorous"
IMG_SIZE = (224, 224)
BATCH_SIZE = 32

print(f"Checking for GPUs... Available: {len(tf.config.list_physical_devices('GPU'))}")


# already split data
# --- 1. PREPARE DATA SPLIT (Automatic) ---
# This creates the train/test/val folders automatically from your uploaded data
def setup_data():
    if os.path.exists(WORKING_DIR):
        shutil.rmtree(WORKING_DIR)
    
    for split in ['train', 'test', 'valid']:
        for cls in ['yes', 'no']:
            os.makedirs(os.path.join(WORKING_DIR, split, cls), exist_ok=True)

    # Split ratios
    splits = {'train': 0.7, 'valid': 0.15, 'test': 0.15}
    
    for cls in ['yes', 'no']:
        src_folder = os.path.join(BASE_DATA_DIR, cls)
        if not os.path.exists(src_folder):
            print(f"ERROR: Could not find {src_folder}")
            return False
            
        files = [f for f in os.listdir(src_folder) if f.endswith('.jpg')]
        np.random.shuffle(files)
        
        n_total = len(files)
        n_train = int(n_total * splits['train'])
        n_val = int(n_total * splits['valid'])
        
        train_files = files[:n_train]
        val_files = files[n_train:n_train+n_val]
        test_files = files[n_train+n_val:]
        
        for f in train_files: shutil.copy(os.path.join(src_folder, f), os.path.join(WORKING_DIR, 'train', cls, f))
        for f in val_files: shutil.copy(os.path.join(src_folder, f), os.path.join(WORKING_DIR, 'valid', cls, f))
        for f in test_files: shutil.copy(os.path.join(src_folder, f), os.path.join(WORKING_DIR, 'test', cls, f))
        
        print(f"Class {cls}: {len(train_files)} Train, {len(val_files)} Val, {len(test_files)} Test")
    return True

# print("Organizing data...")
# if not setup_data():
#     exit()

# --- 2. GENERATORS ---
train_datagen = ImageDataGenerator(
    rescale=1./255, rotation_range=40, width_shift_range=0.4,
    height_shift_range=0.4, shear_range=0.2, zoom_range=0.2,
    horizontal_flip=True, fill_mode='nearest'
)
test_datagen = ImageDataGenerator(rescale=1./255)

train_generator = train_datagen.flow_from_directory(
    os.path.join(WORKING_DIR, 'train'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)
valid_generator = test_datagen.flow_from_directory(
    os.path.join(WORKING_DIR, 'valid'),
    target_size=IMG_SIZE, batch_size=BATCH_SIZE, class_mode='categorical'
)

# --- 3. MODEL ---
# base_model = VGG19(input_shape=IMG_SIZE + (3,), include_top=False, weights='imagenet')
local_weights_path = os.path.expanduser("~/brain_project_data/vgg19_weights_tf_dim_ordering_tf_kernels_notop.h5")
print(f"Loading VGG19 with local weights from: {local_weights_path}")
# Initialize with weights=None (so it doesn't try to download)
base_model = VGG19(
    input_shape=IMG_SIZE + (3,),
    include_top=False,
    weights=None
)
# Load your specific file
base_model.load_weights(local_weights_path)

# Freeze layers
for layer in base_model.layers:
    layer.trainable = False

# Add custom head
x = Flatten()(base_model.output)
x = Dense(4608, activation='relu')(x)
x = Dropout(0.2)(x)
x = Dense(1152, activation='relu')(x)
output = Dense(2, activation='softmax')(x)

model = Model(inputs=base_model.input, outputs=output)

sgd = SGD(learning_rate=0.0001, momentum=0.9, nesterov=True)
model.compile(optimizer=sgd, loss='categorical_crossentropy', metrics=['accuracy'])

# --- 4. TRAIN ---
callbacks = [
    EarlyStopping(monitor='val_loss', patience=4, verbose=1, mode='min'),
    ModelCheckpoint('best_brain_model.keras', monitor='val_loss', save_best_only=True, verbose=1),
    ReduceLROnPlateau(monitor='val_accuracy', factor=0.5, patience=2, verbose=1)
]

print("Starting training on GPU...")
history = model.fit(
    train_generator,
    epochs=200, # You can increase this now because the GPU is fast!
    validation_data=valid_generator,
    callbacks=callbacks
)

# --- 5. PLOT RESULTS ---
print("Plotting results...")

# Ensure we have history data
if 'accuracy' in history.history:
    fig, (ax1, ax2) = plt.subplots(nrows=1, ncols=2, figsize=(12, 6))
    fig.suptitle("Model Training (Frozen CNN)", fontsize=12)
    
    # Get range of epochs actually trained
    max_epoch = len(history.history['accuracy']) + 1
    epochs_list = list(range(1, max_epoch))

    # Plot Accuracy
    ax1.plot(epochs_list, history.history['accuracy'], color='b', linestyle='-', label='Training Data')
    ax1.plot(epochs_list, history.history['val_accuracy'], color='r', linestyle='-', label='Validation Data')
    ax1.set_title('Training Accuracy', fontsize=12)
    ax1.set_xlabel('Epochs', fontsize=12)
    ax1.set_ylabel('Accuracy', fontsize=12)
    ax1.legend(frameon=False, loc='lower center', ncol=2)

    # Plot Loss
    ax2.plot(epochs_list, history.history['loss'], color='b', linestyle='-', label='Training Data')
    ax2.plot(epochs_list, history.history['val_loss'], color='r', linestyle='-', label='Validation Data')
    ax2.set_title('Training Loss', fontsize=12)
    ax2.set_xlabel('Epochs', fontsize=12)
    ax2.set_ylabel('Loss', fontsize=12)
    ax2.legend(frameon=False, loc='upper center', ncol=2)

    # Save the file to disk
    save_path = "training_frozencnn.jpeg"
    plt.savefig(save_path, format='jpeg', dpi=100, bbox_inches='tight')
    print(f"Graph saved to: {save_path}")
    
    # Do NOT call plt.show() on a server!
print("Training Finished. Model saved as 'best_brain_model.keras'")