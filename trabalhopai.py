# oasis-alzheimer-s-detection-binary.py

# --- Import libraries ---
import numpy as np 
import os
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, StratifiedKFold
from keras.models import Sequential
from keras.layers import Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
import nibabel as nib
from skimage.transform import resize
from sklearn.preprocessing import LabelEncoder

# --- Import and Preprocess Dataset ---

# Read the demographic data
demographics_df = pd.read_csv('oasis_longitudinal_demographic.csv', sep=';')

# Clean 'CDR' and 'SES' columns
demographics_df['CDR'] = demographics_df['CDR'].astype(str).str.replace(',', '.', regex=False)
demographics_df['CDR'] = pd.to_numeric(demographics_df['CDR'], errors='coerce')
demographics_df['SES'] = pd.to_numeric(demographics_df['SES'], errors='coerce')

# Define binary diagnosis function
def get_diagnosis_binary(row):
    if row['Group'] == 'Nondemented':
        return 'NonDemented'
    elif row['Group'] == 'Demented':
        return 'Demented'
    elif row['Group'] == 'Converted':
        if row['CDR'] == 0:
            return 'NonDemented'
        elif row['CDR'] > 0:
            return 'Demented'
    return 'Unknown'

demographics_df['Diagnosis'] = demographics_df.apply(get_diagnosis_binary, axis=1)

# Filter out 'Unknown' diagnoses and missing data
demographics_df = demographics_df[demographics_df['Diagnosis'] != 'Unknown'].dropna(subset=['MRI ID', 'Subject ID'])


# --- Patient-Based Train/Test Split ---

# Get unique patients and their final diagnosis (most severe)
patient_diagnoses = demographics_df.groupby('Subject ID')['CDR'].max().reset_index()

def get_final_diagnosis(cdr):
    return 'Demented' if cdr > 0 else 'NonDemented'

patient_diagnoses['Diagnosis'] = patient_diagnoses['CDR'].apply(get_final_diagnosis)

# Encode diagnoses for stratification
le = LabelEncoder()
patient_diagnoses['Diagnosis_encoded'] = le.fit_transform(patient_diagnoses['Diagnosis'])

# Split patients into training (80%) and testing (20%)
train_patients, test_patients = train_test_split(
    patient_diagnoses, 
    test_size=0.2, 
    random_state=42, 
    stratify=patient_diagnoses['Diagnosis_encoded']
)

# Further split training patients into training (80%) and validation (20%)
train_patients, val_patients = train_test_split(
    train_patients,
    test_size=0.2,
    random_state=42,
    stratify=train_patients['Diagnosis_encoded']
)

# Get the MRI IDs for each set
train_mri_ids = demographics_df[demographics_df['Subject ID'].isin(train_patients['Subject ID'])]['MRI ID'].tolist()
val_mri_ids = demographics_df[demographics_df['Subject ID'].isin(val_patients['Subject ID'])]['MRI ID'].tolist()
test_mri_ids = demographics_df[demographics_df['Subject ID'].isin(test_patients['Subject ID'])]['MRI ID'].tolist()


# Create mappings from MRI ID to Diagnosis (0 for NonDemented, 1 for Demented)
demographics_df['Diagnosis_encoded'] = le.transform(demographics_df['Diagnosis'])
mri_diagnosis_map = dict(zip(demographics_df['MRI ID'], demographics_df['Diagnosis_encoded']))


# --- Prepare File Paths and Labels ---

def get_paths_and_labels(mri_ids, mri_map):
    paths = []
    labels = []
    axl_dir = 'axl'
    
    for filename in os.listdir(axl_dir):
        if filename.endswith('.nii'):
            mri_id = filename.replace('_axl.nii', '')
            if mri_id in mri_ids and mri_id in mri_map:
                paths.append(os.path.join(axl_dir, filename))
                labels.append(mri_map[mri_id])
    return paths, labels

x_train_paths, y_train_labels = get_paths_and_labels(train_mri_ids, mri_diagnosis_map)
x_val_paths, y_val_labels = get_paths_and_labels(val_mri_ids, mri_diagnosis_map)
x_test_paths, y_test_labels = get_paths_and_labels(test_mri_ids, mri_diagnosis_map)


# --- Data Loading and Processing ---
image_size = (128, 128)

def load_and_process_images(paths):
    """Loads and resizes NIfTI images."""
    data = []
    for path in paths:
        try:
            nifti_img = nib.load(path)
            img_data = nifti_img.get_fdata()

            # Handle 2D images directly
            if img_data.ndim == 2:
                resized_img = resize(img_data, image_size, anti_aliasing=True)
                # Convert to 3 channels for model compatibility
                img_array = np.stack([resized_img, resized_img, resized_img], axis=-1)
                data.append(img_array)
            else:
                print(f"Skipping {path}: Expected 2D image, got {img_data.ndim}D.")
        except Exception as e:
            print(f"Error loading image {path}: {e}")
    return np.array(data)

print("Loading and processing images for training, validation, and testing sets...")
x_train = load_and_process_images(x_train_paths)
x_val = load_and_process_images(x_val_paths)
x_test = load_and_process_images(x_test_paths)

y_train = np.array(y_train_labels)
y_val = np.array(y_val_labels)
y_test = np.array(y_test_labels)

print(f"Training set: {x_train.shape[0]} images")
print(f"Validation set: {x_val.shape[0]} images")
print(f"Testing set: {x_test.shape[0]} images")
print("---")


# --- Creating Model: CNN for Binary Classification ---
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
model.add(Dense(1, activation='sigmoid')) # Binary classification output
          
model.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
          
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
    validation_data=(x_val, y_val)
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


# --- Example Prediction ---
print("Prediction Example")
# Use a random image from the test set
try:
    if len(x_test) > 0:
        idx = np.random.randint(0, len(x_test))
        img_array = x_test[idx]
        true_label = y_test[idx]
        
        # Reshape for prediction
        x = img_array.reshape(1, image_size[0], image_size[1], 3)
        
        # Predict
        res = model.predict(x)[0][0]
        predicted_label = 1 if res > 0.5 else 0
        
        # Display image and result
        plt.figure(figsize=(4, 4))
        imshow(img_array[:, :, 0], cmap='gray') # Show one channel
        plt.title(f"True: {'Demented' if true_label == 1 else 'NonDemented'}, Predicted: {'Demented' if predicted_label == 1 else 'NonDemented'}")
        plt.show()
        
        print(f"Prediction confidence: {res*100:.2f}% Demented")
    else:
        print("Skipping Example Prediction: No test images found.")
except Exception as e:
    print(f"An error occurred during Prediction Example: {e}")
print("---")
