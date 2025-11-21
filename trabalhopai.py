# trabalhopai.py

# --- Import libraries ---
import numpy as np 
import os
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from keras.applications import DenseNet121
import nibabel as nib
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
import xgboost as xgb
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, roc_auc_score, mean_absolute_error

# --- 1. Data Loading Utility ---
image_size = (128, 128)

def load_images(paths):
    """Loads and resizes NIfTI images."""
    data = []
    for path in paths:
        if os.path.exists(path):
            nifti_img = nib.load(path)
            img_data = nifti_img.get_fdata()
            if img_data.ndim == 2:
                resized_img = resize(img_data, image_size, anti_aliasing=True)
                data.append(resized_img)
            else:
                print(f"Skipping {path}: Not a 2D image.")
        else:
            print(f"Skipping {path}: File not found.")
    return np.array(data)

# --- 2. Data Preparation ---

def load_and_prepare_data():
    """
    Loads and preprocesses the dataset, splitting it into patient-aware
    training, validation, and test sets for both images and labels.
    Returns:
        (x_train_img, y_train_class, y_train_age), 
        (x_val_img, y_val_class, y_val_age), 
        (x_test_img, y_test_class, y_test_age),
        valid_data_df, # Added return
        test_patients # Added return
    """
    # Read and clean the demographic data
    demographics_df = pd.read_csv('oasis_longitudinal_demographic.csv', sep=';')
    demographics_df['CDR'] = demographics_df['CDR'].astype(str).str.replace(',', '.', regex=False)
    demographics_df['CDR'] = pd.to_numeric(demographics_df['CDR'], errors='coerce')
    demographics_df['Age'] = pd.to_numeric(demographics_df['Age'], errors='coerce')

    # --- Binary Diagnosis for Classification ---
    def get_diagnosis_binary(row):
        if row['Group'] == 'Nondemented':
            return 'NonDemented'
        elif row['Group'] == 'Demented':
            return 'Demented'
        elif row['Group'] == 'Converted':
            return 'Demented' if row['CDR'] > 0 else 'NonDemented'
        return 'Unknown'

    demographics_df['Diagnosis'] = demographics_df.apply(get_diagnosis_binary, axis=1)
    
    # Filter out unknown diagnoses and missing data
    valid_data_df = demographics_df[demographics_df['Diagnosis'] != 'Unknown'].dropna(subset=['MRI ID', 'Subject ID', 'Age'])

    # --- Patient-Based Stratified Split ---
    patient_info = valid_data_df.groupby('Subject ID').agg(
        final_CDR=('CDR', 'max'),
        first_visit_age=('Age', 'min')
    ).reset_index()
    
    patient_info['Diagnosis'] = patient_info['final_CDR'].apply(lambda cdr: 'Demented' if cdr > 0 else 'NonDemented')
    
    le = LabelEncoder()
    patient_info['Diagnosis_encoded'] = le.fit_transform(patient_info['Diagnosis'])
    
    # Split patients
    train_val_patients, test_patients = train_test_split(
        patient_info,
        test_size=0.2,
        random_state=42,
        stratify=patient_info['Diagnosis_encoded']
    )
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=0.2, # 20% of the 80% train_val set
        random_state=42,
        stratify=train_val_patients['Diagnosis_encoded']
    )

    # --- Get Image Paths and Labels for each set ---
    def get_set_data(patient_ids_df, full_df):
        set_df = full_df[full_df['Subject ID'].isin(patient_ids_df['Subject ID'])]
        paths = [os.path.join('axl', f"{mri_id}_axl.nii") for mri_id in set_df['MRI ID']]
        class_labels = le.transform(set_df['Diagnosis'])
        age_labels = set_df['Age'].values
        return paths, class_labels, age_labels

    train_paths, y_train_class, y_train_age = get_set_data(train_patients, valid_data_df)
    val_paths, y_val_class, y_val_age = get_set_data(val_patients, valid_data_df)
    test_paths, y_test_class, y_test_age = get_set_data(test_patients, valid_data_df)

    print("Loading images...")
    x_train_img = load_images(train_paths)
    x_val_img = load_images(val_paths)
    x_test_img = load_images(test_paths)
    
    return (x_train_img, y_train_class, y_train_age), (x_val_img, y_val_class, y_val_age), (x_test_img, y_test_class, y_test_age), valid_data_df, test_patients


# --- 3. Feature Extraction ---

def extract_features(images):
    """Extracts texture features from a list of images."""
    features = []
    for img in images:
        img_8bit = (img * 255).astype(np.uint8)
        glcm = graycomatrix(img_8bit, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        props = [graycoprops(glcm, prop)[0, 0] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
        features.append(props)
    return np.array(features)


# --- 4. Evaluation Utilities ---

def evaluate_classifier(y_true, y_pred, model_name):
    """Calculates and prints classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred) # Recall is sensitivity
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("Confusion Matrix:")
    sns.heatmap(cm, annot=True, fmt='d').set(title=f'{model_name} Confusion Matrix')
    plt.show()
    print("----------------------------------\n")

def plot_learning_curves(history, model_name):
    """Plots accuracy learning curves."""
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

def plot_mri_image(mri_id):
    """
    Loads and plots all slices from a .nii or .nii.gz MRI file.
    
    Parameters
    ----------
    filepath : str
        Path to the .nii or .nii.gz file.
    """
    set_df = pd.read_csv('oasis_longitudinal_demographic.csv', sep=';')
    paths = [os.path.join('axl', f"{mri_id}_axl.nii") for mri_id in set_df['MRI ID']]
    # Load the NIfTI file
    filepath=paths[0]
    img = nib.load(filepath)
    data = img.get_fdata()

    # Ensure data is 3D
    img = nib.load(filepath)
    data = img.get_fdata()
    header = img.header
    affine = img.affine

    # ================================
    # PRINT METADATA
    # ================================
    print("\n=== MRI METADATA ===")
    print(f"File: {filepath}")
    print(f"Shape: {data.shape}")
    print(f"Data Type: {header.get_data_dtype()}")
    print(f"Voxel Dimensions (pixdim): {header['pixdim'][1:4]}")
    print("\n--- Full Header ---")
    print(header)   # prints ALL metadata fields

    print("\n--- Affine Matrix (qform/sform) ---")
    print(affine)

    # -----------------------------
    # CASE 1 → 2D IMAGE (single slice)
    # -----------------------------
    if data.ndim == 2:
        plt.figure(figsize=(6, 6))
        plt.imshow(data, cmap='gray')
        plt.title(f"{mri_id} - Single Slice (2D)")
        plt.axis("off")
        plt.show()
        return


    
    # -----------------------------
    # CASE 2 → 3D IMAGE (multiple slices)
    # -----------------------------
    if data.ndim == 3:
        num_slices = data.shape[2]

        cols = 8
        rows = int(np.ceil(num_slices / cols))

        plt.figure(figsize=(15, rows * 2))

        for i in range(num_slices):
            plt.subplot(rows, cols, i + 1)
            plt.imshow(data[:, :, i], cmap='gray')
            plt.axis("off")

        plt.suptitle(f"{mri_id} - All Slices")
        plt.tight_layout()
        plt.show()
        return

    # -----------------------------
    # OTHER CASES (rare)
    # -----------------------------
    raise ValueError(f"Unexpected MRI shape: {data.shape}")

# --- Main Execution ---

if __name__ == '__main__':
    plot_mri_image("OAS2_0001")
    # --- Data Loading ---
    (x_train_img, y_train_class, y_train_age), \
    (x_val_img, y_val_class, y_val_age), \
    (x_test_img, y_test_class, y_test_age), \
    valid_data_df, test_patients = load_and_prepare_data() # Added return values

    print("\n--- TASK 1: CLASSIFICATION (Demented vs. NonDemented) ---")
    
    # --- Feature Extraction and Scaling for Shallow Models ---
    print("Extracting features for shallow classification models...")
    x_train_features = extract_features(x_train_img)
    x_test_features = extract_features(x_test_img)
    print(f"Feature extraction complete. Feature shape: {x_train_features.shape}")

    print("Scaling features...")
    scaler = StandardScaler()
    x_train_features_scaled = scaler.fit_transform(x_train_features)
    x_test_features_scaled = scaler.transform(x_test_features)
    
    # --- Shallow Model: SVM Classifier ---
    print("\nTraining SVM Classifier...")
    svm_classifier = SVC(kernel='rbf', probability=True, random_state=42)
    svm_classifier.fit(x_train_features_scaled, y_train_class)
    y_pred_svm = svm_classifier.predict(x_test_features_scaled)
    evaluate_classifier(y_test_class, y_pred_svm, "SVM Classifier")

    # --- Deep Model: DenseNet Classifier ---
    x_train_deep = np.stack([x_train_img, x_train_img, x_train_img], axis=-1)
    x_val_deep = np.stack([x_val_img, x_val_img, x_val_img], axis=-1)
    x_test_deep = np.stack([x_test_img, x_test_img, x_test_img], axis=-1)

    print("\nCreating and fine-tuning DenseNet Classifier...")
    base_model_clf = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model_clf.trainable = False # Freeze base
    
    x = Flatten()(base_model_clf.output)
    x = Dense(512, activation='relu')(x)
    x = Dropout(0.5)(x)
    output_clf = Dense(1, activation='sigmoid')(x)
    
    model_clf = Model(inputs=base_model_clf.input, outputs=output_clf)
    model_clf.compile(loss='binary_crossentropy', optimizer='Adamax', metrics=['accuracy'])
    
    history_clf = model_clf.fit(
        x_train_deep, y_train_class,
        validation_data=(x_val_deep, y_val_class),
        epochs=10, batch_size=16, verbose=1
    )
    
    plot_learning_curves(history_clf, "DenseNet Classifier")
    y_pred_deep_clf_proba = model_clf.predict(x_test_deep)
    y_pred_deep_clf = (y_pred_deep_clf_proba > 0.5).astype(int)
    evaluate_classifier(y_test_class, y_pred_deep_clf, "DenseNet Classifier")


    print("\n--- TASK 2: REGRESSION (Predicting Patient Age) ---")

    # --- Shallow Model: XGBoost Regressor ---
    print("\nTraining XGBoost Regressor...")
    xgb_regressor = xgb.XGBRegressor(objective='reg:squarederror', random_state=42)
    # Note: Using scaled features as well for the regressor
    xgb_regressor.fit(x_train_features_scaled, y_train_age)
    y_pred_age_xgb = xgb_regressor.predict(x_test_features_scaled)
    mae_xgb = mean_absolute_error(y_test_age, y_pred_age_xgb)
    print(f"--- XGBoost Regressor Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae_xgb:.4f} years")
    print("------------------------------------\n")

    # --- Deep Model: DenseNet Regressor ---
    print("\nCreating and fine-tuning DenseNet Regressor...")
    base_model_reg = DenseNet121(weights='imagenet', include_top=False, input_shape=(128, 128, 3))
    base_model_reg.trainable = False
    
    x_reg = Flatten()(base_model_reg.output)
    x_reg = Dense(512, activation='relu')(x_reg)
    x_reg = Dropout(0.5)(x_reg)
    output_reg = Dense(1, activation='linear')(x_reg) # Linear activation for regression
    
    model_reg = Model(inputs=base_model_reg.input, outputs=output_reg)
    model_reg.compile(loss='mean_squared_error', optimizer='Adamax', metrics=['mae'])
    
    history_reg = model_reg.fit(
        x_train_deep, y_train_age,
        validation_data=(x_val_deep, y_val_age),
        epochs=10, batch_size=16, verbose=1
    )
    
    y_pred_age_deep = model_reg.predict(x_test_deep).flatten()
    mae_deep = mean_absolute_error(y_test_age, y_pred_age_deep)
    print(f"--- DenseNet Regressor Evaluation ---")
    print(f"Mean Absolute Error (MAE): {mae_deep:.4f} years")
    print("-------------------------------------\n")

    # --- Analysis Questions ---
    print("\n--- Analysis Questions ---")
    print("1. Are the inputs sufficient for a good prediction?")
    print(f"For age prediction, the shallow model (XGBoost) had an MAE of {mae_xgb:.2f} years, and the deep model (DenseNet) had an MAE of {mae_deep:.2f} years. This indicates the models have some predictive power, but there's room for improvement. The image features seem to provide a reasonable but not perfect estimation of age.")
    
    print("\n2. Do later visits result in higher predicted ages?")
    
    # Re-fetch the test set data needed for this analysis, now that valid_data_df and test_patients are available
    test_analysis_df = valid_data_df[valid_data_df['Subject ID'].isin(test_patients['Subject ID'])].copy()
    test_analysis_paths = [os.path.join('axl', f"{mri_id}_axl.nii") for mri_id in test_analysis_df['MRI ID']]
    x_test_img_analysis = load_images(test_analysis_paths)
    x_test_features_analysis = extract_features(x_test_img_analysis)
    x_test_features_analysis_scaled = scaler.transform(x_test_features_analysis) # Use the same scaler fitted on training data
    
    test_analysis_df['predicted_age_xgb'] = xgb_regressor.predict(x_test_features_analysis_scaled)
    
    consistent_increase = 0
    total_patients_mv = 0
    
    # Sort by Subject ID and then by Visit to ensure correct order for diff
    for subject_id, group in test_analysis_df.sort_values(['Subject ID', 'Visit']).groupby('Subject ID'):
        if len(group) > 1: # Only consider patients with multiple visits
            total_patients_mv += 1
            predicted_ages = group['predicted_age_xgb'].values
            if np.all(np.diff(predicted_ages) >= 0):
                consistent_increase += 1
    
    if total_patients_mv > 0:
        consistency_ratio = consistent_increase / total_patients_mv
        print(f"For the XGBoost regressor, {consistency_ratio:.2%} of patients with multiple visits showed consistently non-decreasing predicted ages.")
    else:
        print("Not enough data with multiple visits in the test set to perform this analysis.")
    print("--------------------------")