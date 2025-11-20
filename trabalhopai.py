# oasis-alzheimer-s-detection.py

# --- Import libraries ---
import numpy as np 
import os
import keras
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from keras.models import Sequential
from PIL import Image
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from sklearn.preprocessing import OneHotEncoder
from sklearn.model_selection import train_test_split
from matplotlib.pyplot import imshow

# --- Import Dataset ---
# NOTE: This code assumes the dataset structure exists at the specified paths.
# If you run this script outside of the Kaggle environment it was designed for, 
# you'll need to modify the base path '/kaggle/input/imagesoasis/Data/' 
# to where your data is located.

path1 = [] # Non Demented
path2 = [] # Mild Dementia
path3 = [] # Moderate Dementia
path4 = [] # Very mild Dementia

for dirname, _, filenames in os.walk('/kaggle/input/imagesoasis/Data/Non Demented'):
    for filename in filenames:
        path1.append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk('/kaggle/input/imagesoasis/Data/Mild Dementia'):
    for filename in filenames:
        path2.append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk('/kaggle/input/imagesoasis/Data/Moderate Dementia'):
    for filename in filenames:
        path3.append(os.path.join(dirname, filename))
        
for dirname, _, filenames in os.walk('/kaggle/input/imagesoasis/Data/Very mild Dementia'):
    for filename in filenames:
        path4.append(os.path.join(dirname, filename))

# Limiting the number of files to the first 100 in each category (as done in the notebook)
path1 = path1[0:100]
path2 = path2[0:100]
path3 = path3[0:100]
path4 = path4[0:100]


# --- One Hot Encoding ---
encoder = OneHotEncoder()
encoder.fit([[0],[1],[2],[3]])

# 0 --> Non Demented
# 1 --> Mild Dementia
# 2 --> Moderate Dementia
# 3 --> Very Mild Dementia

print("OneHotEncoder initialized and fitted.")
print("---")

# --- Data Loading, Resizing, and One-Hot Encoding ---
data = []
result = []
image_size = (128, 128)

def load_and_process_images(path_list, label):
    """Loads, resizes, and converts images to numpy arrays, and applies one-hot encoding."""
    count = 0
    for path in path_list:
        try:
            img = Image.open(path).convert('RGB') # Ensure 3 color channels
            img = img.resize(image_size)
            img_array = np.array(img)
            
            # Check for the expected 3-channel shape (128, 128, 3)
            if img_array.shape == (image_size[0], image_size[1], 3):
                data.append(img_array)
                # Apply one-hot encoding for the label (0, 1, 2, or 3)
                encoded_label = encoder.transform([[label]]).toarray()
                result.append(encoded_label)
                count += 1
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return count

print("Loading and processing images...")
count1 = load_and_process_images(path1, 0)
count2 = load_and_process_images(path2, 1)
count3 = load_and_process_images(path3, 2)
count4 = load_and_process_images(path4, 3)
print(f"Loaded: Non Demented ({count1}), Mild Dementia ({count2}), Moderate Dementia ({count3}), Very Mild Dementia ({count4})")

# Convert lists to numpy arrays
data = np.array(data)
result = np.array(result)
result = result.reshape((data.shape[0], 4)) # Reshape the result array

print(f"Data shape: {data.shape}")
print(f"Result (Labels) shape: {result.shape}")
print("---")

# --- Splitting The Data ---
x_train, x_test, y_train, y_test = train_test_split(data, result, test_size=0.15, shuffle=True, random_state=42)

print("Data split into training and testing sets.")
print(f"x_train shape: {x_train.shape}")
print(f"y_train shape: {y_train.shape}")
print("---")

# --- Creating Model: CNN ---
print("Creating and compiling the CNN model...")
model = Sequential()

# First Convolutional Block
model.add(Conv2D(32, kernel_size=(2, 2), input_shape=(image_size[0], image_size[1], 3), padding='same'))
model.add(Conv2D(32, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))

# Second Convolutional Block
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(Conv2D(64, kernel_size=(2, 2), activation='relu', padding='same'))
model.add(BatchNormalization())
model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))
model.add(Dropout(0.25))

# Classifier Block
model.add(Flatten())
model.add(Dense(512, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(4, activation='softmax'))
          
model.compile(loss='categorical_crossentropy', optimizer='Adamax', metrics=['accuracy'])
          
model.summary()
print("---")

# --- Model Training ---
print("Starting model training...")
history = model.fit(
    x_train, 
    y_train, 
    epochs=10, 
    batch_size=10, 
    verbose=1, 
    validation_data=(x_test, y_test)
)
print("Model training complete.")
print("---")

# --- Plot Model Loss ---
plt.figure(figsize=(8, 5))
plt.plot(history.history['loss'], label='Training Loss')
plt.plot(history.history['val_loss'], label='Validation Loss')
plt.title('Model Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend(['Training', 'Validation'], loc='upper right')
plt.show()
print("Loss plot generated.")
print("---")

# --- Accuracy and Prediction Helper Function ---
def names(number):
    """Maps the numeric label (0-3) to the corresponding dementia stage name."""
    if number == 0:
        return 'Non Demented'
    elif number == 1:
        return 'Mild Dementia'
    elif number == 2:
        return 'Moderate Dementia'
    elif number == 3:
        return 'Very Mild Dementia'
    else:
        return 'Error in Prediction'

# --- Example Prediction 1 ---
print("Prediction Example 1: Moderate Dementia")
# The original notebook used a specific image: 'Moderate Dementia/OAS1_0308_MR1_mpr-1_101.jpg'
# To make this runnable, we will try to select a random image from the loaded data for "Moderate Dementia" (index 200-299)
# We will use the first one available in the loaded path list (path3[0])
try:
    if path3:
        image_path = path3[0]
        img = Image.open(image_path).convert('RGB')
        x = np.array(img.resize(image_size))
        x = x.reshape(1, image_size[0], image_size[1], 3)
        
        # Predict
        res = model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        
        # Display image and result
        plt.figure(figsize=(4, 4))
        imshow(img)
        plt.title(f"Predicted: {names(classification)}")
        plt.show()
        
        print(f"{res[0][classification]*100:.4f}% Confidence This Is {names(classification)}")
    else:
        print("Skipping Example 1: No 'Moderate Dementia' images found in path.")
except Exception as e:
    print(f"An error occurred during Prediction Example 1: {e}")
print("---")


# --- Example Prediction 2 ---
print("Prediction Example 2: Very mild Dementia")
# The original notebook used a specific image: 'Very mild Dementia/OAS1_0003_MR1_mpr-1_117.jpg'
# We will use the first one available in the loaded path list for "Very mild Dementia" (path4[0])
try:
    if path4:
        image_path = path4[0]
        img = Image.open(image_path).convert('RGB')
        x = np.array(img.resize(image_size))
        x = x.reshape(1, image_size[0], image_size[1], 3)
        
        # Predict
        res = model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        
        # Display image and result
        plt.figure(figsize=(4, 4))
        imshow(img)
        plt.title(f"Predicted: {names(classification)}")
        plt.show()
        
        print(f"{res[0][classification]*100:.4f}% Confidence This Is {names(classification)}")
    else:
        print("Skipping Example 2: No 'Very mild Dementia' images found in path.")
except Exception as e:
    print(f"An error occurred during Prediction Example 2: {e}")
print("---")


# --- Example Prediction 3 ---
print("Prediction Example 3: Mild Dementia")
# The original notebook used a specific image: 'Mild Dementia/OAS1_0028_MR1_mpr-1_145.jpg'
# We will use the first one available in the loaded path list for "Mild Dementia" (path2[0])
try:
    if path2:
        image_path = path2[0]
        img = Image.open(image_path).convert('RGB')
        x = np.array(img.resize(image_size))
        x = x.reshape(1, image_size[0], image_size[1], 3)
        
        # Predict
        res = model.predict_on_batch(x)
        classification = np.where(res == np.amax(res))[1][0]
        
        # Display image and result
        plt.figure(figsize=(4, 4))
        imshow(img)
        plt.title(f"Predicted: {names(classification)}")
        plt.show()
        
        print(f"{res[0][classification]*100:.4f}% Confidence This Is {names(classification)}")
    else:
        print("Skipping Example 3: No 'Mild Dementia' images found in path.")
except Exception as e:
    print(f"An error occurred during Prediction Example 3: {e}")
print("---")