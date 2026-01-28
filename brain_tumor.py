cd .\braintumor  
.\brain_env\Scripts\activate


!.\brain_env\Scripts\activate
pip install ipykernel
python -m ipykernel install --user --name=brain_env --display-name "Python (Brain Tumor Project)"
pip install pandas numpy matplotlib opencv-python seaborn

#import library
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import os, shutil
import cv2
import matplotlib.image as mpimg
import seaborn as sns
%matplotlib inline
plt.style.use('ggplot')

folder = 'yes/'
count = 1

for filename in os.listdir(folder):
    source = folder + filename
    destination = folder + "Y_" +str(count)+".jpg"
    os.rename(source, destination)
    count+=1
print("All files are renamed in the yes dir.")

folder = 'no/'
count = 1

for filename in os.listdir(folder):
    source = folder + filename
    destination = folder + "N_" +str(count)+".jpg"
    os.rename(source, destination)
    count+=1
print("All files are renamed in the no dir.")


import os
import cv2

yes_path = "yes/"
no_path = "no/"

os.makedirs(yes_path, exist_ok=True)
os.makedirs(no_path, exist_ok=True)

def count_real_jpg(folder_path):
    count = 0
    for f in os.listdir(folder_path):
        if f.lower().endswith(".jpg"):
            full_path = os.path.join(folder_path, f)
            img = cv2.imread(full_path)
            if img is not None:
                count += 1
    return count

number_files_yes = count_real_jpg(yes_path)
print(number_files_yes)

number_files_no = count_real_jpg(no_path)
print(number_files_no)

# for 115: 98 yes/no imbalance, we will use augmentation on the no part to deal with the imbalance 
pip install tensorflow 

import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications.vgg19 import VGG19
from tensorflow.keras.optimizers import SGD, Adam
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau

# timing block
def timing(sec_elapsed):
    h = int(sec_elapsed / (60*60))
    m = int(sec_elapsed % (60*60) / 60)
    s = sec_elapsed % 60
    return f"{h}:{m}:{s}"

# random rotae 10 degrees, draken, move, flip the image, 
# fill the color of nearest pixel for empty space created by rotation
# since the tumor can appear in either side, flip the image should be fine 
def augmented_data(file_dir, n_generated_samples, save_to_dir):
    data_gen = ImageDataGenerator(
        rotation_range = 10,
        width_shift_range = 0.1,
        height_shift_range=0.1,
        shear_range=0.1,
        brightness_range=(0.3, 1.0),
        horizontal_flip=True,
        vertical_flip=True,
        fill_mode='nearest'
    )
    os.makedirs(save_to_dir, exist_ok=True)
    # only check 1 image vector printed out by setting a flag 
    first_img_flag = True 
    for filename in os.listdir(file_dir):
        image_path = os.path.join(file_dir, filename)
        # cv2.imread: read image into original resolution and RGB color mode (3 channels)
        image = cv2.imread(image_path)

        if image is None:
            continue
        image = image.reshape((1,)+image.shape)
        save_prefix ="aug_" + filename[:-4]

        i=0
        for batch in data_gen.flow(
            x=image,
            batch_size =1,
            save_to_dir=save_to_dir,
            save_prefix=save_prefix,
            save_format="jpg"
        ):
            if first_img_flag:
                print("--- EXAMPLE VECTOR (Showing only once) ---")
                print(f"Shape: {batch.shape}")
                print(f"First pixel: {batch[0,0,0,:]}")
                first_img_flag = False  # <--- 3. Turn it off immediately
            i+=1
            if n > generated_samples:
                break

import time
start_time = time.time()

yes_path = 'yes'
no_path = 'no'

augmented_data_path = 'augmented_data/'

augmented_data(file_dir = yes_path, n_generated_samples=6, save_to_dir=augmented_data_path+'yes')
augmented_data(file_dir = no_path, n_generated_samples=9, save_to_dir=augmented_data_path+'no')

end_time = time.time()
execution_time = end_time - start_time
print(timing(execution_time))

