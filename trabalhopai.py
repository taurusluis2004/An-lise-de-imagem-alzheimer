# trabalhopai.py - VERSÃO COMPLETA COM INTERFACE GRÁFICA
# Sistema de Análise de Imagens - Alzheimer

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np 
import os
import pandas as pd 
import seaborn as sns
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, mean_absolute_error
import xgboost as xgb
import nibabel as nib
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from keras.models import Model
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
from keras.applications import DenseNet121

# =============================================================================
# CLASSE IMAGE LOADER (integrado de image_loader.py)
# =============================================================================
class ImageLoader:
    """Classe para carregar imagens médicas em diferentes formatos."""
    
    @staticmethod
    def load_image(file_path):
        """
        Carrega uma imagem de qualquer formato suportado.
        
        Args:
            file_path (str): Caminho para o arquivo de imagem
            
        Returns:
            tuple: (image_data, metadata)
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        ext = os.path.splitext(file_path.lower())[1]
        
        if file_path.lower().endswith('.nii.gz'):
            return ImageLoader._load_nifti(file_path)
        elif ext in ['.nii']:
            return ImageLoader._load_nifti(file_path)
        elif ext in ['.png', '.jpg', '.jpeg']:
            return ImageLoader._load_2d_image(file_path)
        else:
            raise ValueError(f"Formato de arquivo não suportado: {ext}")
    
    @staticmethod
    def _load_nifti(file_path):
        """Carrega imagem Nifti (3D ou 4D)."""
        img = nib.load(file_path)
        data = img.get_fdata()
        
        metadata = {
            'format': 'nifti',
            'shape': data.shape,
            'dimensions': len(data.shape),
            'affine': img.affine,
            'header': img.header,
            'voxel_sizes': img.header.get_zooms(),
            'file_path': file_path
        }
        
        return data, metadata
    
    @staticmethod
    def _load_2d_image(file_path):
        """Carrega imagem 2D (PNG, JPG)."""
        img = Image.open(file_path)
        data = np.array(img)
        
        metadata = {
            'format': 'image2d',
            'shape': data.shape,
            'dimensions': len(data.shape),
            'mode': img.mode,
            'size': img.size,
            'file_path': file_path
        }
        
        return data, metadata
    
    @staticmethod
    def get_slice(image_data, axis=2, slice_idx=None):
        """Extrai uma fatia 2D de uma imagem 3D."""
        if len(image_data.shape) == 2:
            return image_data
        
        if len(image_data.shape) == 3:
            if slice_idx is None:
                slice_idx = image_data.shape[axis] // 2
            
            if axis == 0:
                return image_data[slice_idx, :, :]
            elif axis == 1:
                return image_data[:, slice_idx, :]
            elif axis == 2:
                return image_data[:, :, slice_idx]
        
        raise ValueError(f"Não é possível extrair fatia de imagem com shape {image_data.shape}")
    
    @staticmethod
    def normalize_for_display(image_data):
        """Normaliza dados da imagem para exibição (0-255)."""
        data = image_data.copy()
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        data_min = data.min()
        data_max = data.max()
        
        if data_max > data_min:
            data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
        
        return data

# =============================================================================
# FUNÇÕES DE PROCESSAMENTO DE DADOS
# =============================================================================
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

def load_and_prepare_data():
    """Loads and preprocesses the dataset."""
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'oasis_longitudinal_demographic.csv')
    demographics_df = pd.read_csv(csv_path, sep=';')
    demographics_df['CDR'] = demographics_df['CDR'].astype(str).str.replace(',', '.', regex=False)
    demographics_df['CDR'] = pd.to_numeric(demographics_df['CDR'], errors='coerce')
    demographics_df['Age'] = pd.to_numeric(demographics_df['Age'], errors='coerce')

    def get_diagnosis_binary(row):
        if row['Group'] == 'Nondemented':
            return 'NonDemented'
        elif row['Group'] == 'Demented':
            return 'Demented'
        elif row['Group'] == 'Converted':
            return 'Demented' if row['CDR'] > 0 else 'NonDemented'
        return 'Unknown'

    demographics_df['Diagnosis'] = demographics_df.apply(get_diagnosis_binary, axis=1)
    valid_data_df = demographics_df[demographics_df['Diagnosis'] != 'Unknown'].dropna(subset=['MRI ID', 'Subject ID', 'Age'])

    patient_info = valid_data_df.groupby('Subject ID').agg(
        final_CDR=('CDR', 'max'),
        first_visit_age=('Age', 'min')
    ).reset_index()
    
    patient_info['Diagnosis'] = patient_info['final_CDR'].apply(lambda cdr: 'Demented' if cdr > 0 else 'NonDemented')
    
    le = LabelEncoder()
    patient_info['Diagnosis_encoded'] = le.fit_transform(patient_info['Diagnosis'])
    
    train_val_patients, test_patients = train_test_split(
        patient_info, test_size=0.2, random_state=42,
        stratify=patient_info['Diagnosis_encoded']
    )
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.2, random_state=42,
        stratify=train_val_patients['Diagnosis_encoded']
    )

    def get_set_data(patient_ids_df, full_df):
        set_df = full_df[full_df['Subject ID'].isin(patient_ids_df['Subject ID'])]
        axl_dir = os.path.join(base_dir, 'axl')
        paths = [os.path.join(axl_dir, f"{mri_id}_axl.nii") for mri_id in set_df['MRI ID']]
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
    
    return (x_train_img, y_train_class, y_train_age), \
           (x_val_img, y_val_class, y_val_age), \
           (x_test_img, y_test_class, y_test_age), \
           valid_data_df, test_patients

def extract_features(images):
    """Extracts texture features from a list of images."""
    features = []
    for img in images:
        img_8bit = (img * 255).astype(np.uint8)
        glcm = graycomatrix(img_8bit, distances=[1], angles=[0], levels=256, symmetric=True, normed=True)
        props = [graycoprops(glcm, prop)[0, 0] for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']]
        features.append(props)
    return np.array(features)

def evaluate_classifier(y_true, y_pred, model_name):
    """Calculates and prints classification metrics."""
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel() if cm.shape == (2, 2) else (0, 0, 0, 0)
    
    accuracy = accuracy_score(y_true, y_pred)
    sensitivity = recall_score(y_true, y_pred)
    specificity = tn / (tn + fp) if (tn + fp) > 0 else 0
    
    print(f"--- {model_name} Evaluation ---")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Sensitivity (Recall): {sensitivity:.4f}")
    print(f"Specificity: {specificity:.4f}")
    print("Confusion Matrix:")
    print(cm)
    return accuracy, sensitivity, specificity, cm

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

# =============================================================================
# INTERFACE GRÁFICA
# =============================================================================
class AlzheimerAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Análise de Imagens - Alzheimer")
        self.root.geometry("1200x800")
        
        # Variáveis de controle
        self.current_image_data = None
        self.current_metadata = None
        self.current_slice_index = 0
        self.current_axis = 2  # Axial
        self.zoom_level = 1.0
        
        # Dataset paths
        self.dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'axl')
        
        # Modelos
        self.svm_model = None
        self.xgboost_model = None
        self.densenet_classif = None
        self.densenet_regress = None
        
        # Dados
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.valid_data_df = None
        self.test_patients = None
        
        # Características extraídas
        self.x_train_features = None
        self.x_val_features = None
        self.x_test_features = None
        
        # Acessibilidade - Tamanho de fonte
        self.font_size = 12
        self.base_font = ("Arial", self.font_size)
        
        self.setup_ui()
    
    def setup_ui(self):
        """Configura a interface do usuário"""
        # Estilo
        self.style = ttk.Style()
        self.style.configure("TButton", font=self.base_font, padding=5)
        self.style.configure("TLabel", font=self.base_font)
        
        # Menu Superior
        menu_bar = tk.Menu(self.root)
        
        # Menu Arquivo
        menu_arquivo = tk.Menu(menu_bar, tearoff=0)
        menu_arquivo.add_command(label="Carregar Imagem", command=self.carregar_dataset)
        menu_arquivo.add_command(label="Carregar Arquivo Externo", command=self.carregar_arquivo_externo)
        menu_arquivo.add_separator()
        menu_arquivo.add_command(label="Sair", command=self.root.quit)
        menu_bar.add_cascade(label="Arquivo", menu=menu_arquivo)
        
        # Menu Dados
        menu_dados = tk.Menu(menu_bar, tearoff=0)
        menu_dados.add_command(label="Preparar Dados (80/20)", command=self.preparar_dados)
        menu_dados.add_command(label="Extrair Características", command=self.extrair_caracteristicas)
        menu_bar.add_cascade(label="Dados", menu=menu_dados)
        
        # Menu SVM
        menu_svm = tk.Menu(menu_bar, tearoff=0)
        menu_svm.add_command(label="Treinar SVM", command=self.treinar_svm)
        menu_svm.add_command(label="Avaliar SVM", command=self.avaliar_svm)
        menu_svm.add_command(label="Matriz de Confusão", command=self.matriz_confusao_svm)
        menu_bar.add_cascade(label="SVM", menu=menu_svm)
        
        # Menu XGBoost
        menu_xgb = tk.Menu(menu_bar, tearoff=0)
        menu_xgb.add_command(label="Treinar XGBoost", command=self.treinar_xgboost)
        menu_xgb.add_command(label="Avaliar XGBoost", command=self.avaliar_xgboost)
        menu_xgb.add_command(label="Análise Temporal", command=self.analise_temporal)
        menu_bar.add_cascade(label="XGBoost", menu=menu_xgb)
        
        # Menu DenseNet
        menu_densenet = tk.Menu(menu_bar, tearoff=0)
        menu_densenet.add_command(label="Treinar Classificação", command=self.treinar_densenet_classif)
        menu_densenet.add_command(label="Treinar Regressão", command=self.treinar_densenet_regress)
        menu_densenet.add_separator()
        menu_densenet.add_command(label="Avaliar Classificação", command=self.avaliar_densenet_classif)
        menu_densenet.add_command(label="Avaliar Regressão", command=self.avaliar_densenet_regress)
        menu_bar.add_cascade(label="DenseNet", menu=menu_densenet)
        
        # Menu Acessibilidade
        menu_acess = tk.Menu(menu_bar, tearoff=0)
        menu_acess.add_command(label="Aumentar Fonte", command=self.aumentar_fonte)
        menu_acess.add_command(label="Diminuir Fonte", command=self.diminuir_fonte)
        menu_bar.add_cascade(label="Acessibilidade", menu=menu_acess)
        
        self.root.config(menu=menu_bar)
        
        # Layout Principal
        # Painel Esquerdo
        self.painel_esquerdo = ttk.Frame(self.root, width=300)
        self.painel_esquerdo.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        # Painel Direito
        self.painel_direito = ttk.Frame(self.root)
        self.painel_direito.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        # === PAINEL ESQUERDO ===
        ttk.Label(self.painel_esquerdo, text="Informações da Imagem", 
                 font=("Arial", 14, "bold")).pack(pady=10)
        
        self.info_frame = ttk.LabelFrame(self.painel_esquerdo, text="Dados", padding=10)
        self.info_frame.pack(fill=tk.X, pady=5)
        
        self.lbl_arquivo = ttk.Label(self.info_frame, text="Arquivo: Nenhum carregado")
        self.lbl_arquivo.pack(anchor=tk.W)
        
        self.lbl_formato = ttk.Label(self.info_frame, text="Formato: -")
        self.lbl_formato.pack(anchor=tk.W)
        
        self.lbl_dimensoes = ttk.Label(self.info_frame, text="Dimensões: -")
        self.lbl_dimensoes.pack(anchor=tk.W)
        
        # Controles de navegação
        self.nav_frame = ttk.LabelFrame(self.painel_esquerdo, text="Navegação", padding=10)
        self.nav_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.nav_frame, text="Fatia:").pack(anchor=tk.W, pady=(5, 0))
        self.slice_slider = ttk.Scale(self.nav_frame, from_=0, to=100, orient=tk.HORIZONTAL,
                                     command=self.mudar_fatia)
        self.slice_slider.pack(fill=tk.X)
        
        self.lbl_slice = ttk.Label(self.nav_frame, text="Fatia: 0/0")
        self.lbl_slice.pack(anchor=tk.W)
        
        # Controles de zoom
        self.zoom_frame = ttk.LabelFrame(self.painel_esquerdo, text="Zoom", padding=10)
        self.zoom_frame.pack(fill=tk.X, pady=10)
        
        btn_frame = ttk.Frame(self.zoom_frame)
        btn_frame.pack()
        
        ttk.Button(btn_frame, text="Zoom In (+)", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Zoom Out (-)", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Reset", command=self.zoom_reset).pack(side=tk.LEFT, padx=2)
        
        self.lbl_zoom = ttk.Label(self.zoom_frame, text="Zoom: 100%")
        self.lbl_zoom.pack(pady=5)
        
        # === PAINEL DIREITO ===
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Visualizador de Imagens Médicas")
        self.ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.painel_direito)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        self.ax.text(0.5, 0.5, 'Carregue uma imagem\npelo menu Arquivo', 
                    ha='center', va='center', fontsize=16, transform=self.ax.transAxes)
    
    # =========================================================================
    # FUNÇÕES DE CARREGAMENTO DE IMAGEM
    # =========================================================================
    def carregar_dataset(self):
        """Carrega uma imagem do dataset"""
        if not os.path.exists(self.dataset_path):
            messagebox.showerror("Erro", f"Diretório não encontrado: {self.dataset_path}")
            return
        
        files = [f for f in os.listdir(self.dataset_path) if f.endswith('.nii')]
        if not files:
            messagebox.showerror("Erro", "Nenhuma imagem .nii encontrada")
            return
        
        dialog = tk.Toplevel(self.root)
        dialog.title("Selecionar Imagem")
        dialog.geometry("400x500")
        
        ttk.Label(dialog, text="Imagens Disponíveis:", font=("Arial", 12, "bold")).pack(pady=10)
        
        listbox = tk.Listbox(dialog, font=("Arial", 10))
        listbox.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        for f in sorted(files):
            listbox.insert(tk.END, f)
        
        def selecionar():
            selection = listbox.curselection()
            if selection:
                filename = listbox.get(selection[0])
                filepath = os.path.join(self.dataset_path, filename)
                dialog.destroy()
                self.carregar_e_exibir(filepath)
        
        ttk.Button(dialog, text="Carregar", command=selecionar).pack(pady=10)
    
    def carregar_arquivo_externo(self):
        """Carrega arquivo externo (NIfTI, PNG, JPG)"""
        filetypes = [
            ("Todos os suportados", "*.nii *.nii.gz *.png *.jpg *.jpeg"),
            ("NIfTI", "*.nii *.nii.gz"),
            ("Imagens", "*.png *.jpg *.jpeg")
        ]
        filepath = filedialog.askopenfilename(title="Selecionar Imagem", filetypes=filetypes)
        if filepath:
            self.carregar_e_exibir(filepath)
    
    def carregar_e_exibir(self, filepath):
        """Carrega e exibe uma imagem"""
        try:
            self.current_image_data, self.current_metadata = ImageLoader.load_image(filepath)
            
            self.lbl_arquivo.config(text=f"Arquivo: {os.path.basename(filepath)}")
            self.lbl_formato.config(text=f"Formato: {self.current_metadata['format']}")
            self.lbl_dimensoes.config(text=f"Dimensões: {self.current_metadata['shape']}")
            
            if len(self.current_metadata['shape']) == 3:
                max_slice = self.current_metadata['shape'][self.current_axis] - 1
                self.slice_slider.config(to=max_slice)
                self.current_slice_index = max_slice // 2
                self.slice_slider.set(self.current_slice_index)
            else:
                self.slice_slider.config(to=0)
                self.current_slice_index = 0
            
            self.zoom_level = 1.0
            self.exibir_imagem()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem:\n{str(e)}")
    
    def exibir_imagem(self):
        """Exibe a imagem atual no canvas"""
        if self.current_image_data is None:
            return
        
        self.ax.clear()
        
        if len(self.current_image_data.shape) == 3:
            slice_data = ImageLoader.get_slice(self.current_image_data, 
                                              self.current_axis, 
                                              self.current_slice_index)
            max_slice = self.current_image_data.shape[self.current_axis] - 1
            self.lbl_slice.config(text=f"Fatia: {self.current_slice_index}/{max_slice}")
        else:
            slice_data = self.current_image_data
        
        display_data = ImageLoader.normalize_for_display(slice_data)
        
        self.ax.imshow(display_data, cmap='gray')
        self.ax.set_title(f"{os.path.basename(self.current_metadata['file_path'])} - Zoom: {int(self.zoom_level*100)}%")
        self.ax.axis('off')
        
        if self.zoom_level != 1.0:
            xlim = self.ax.get_xlim()
            ylim = self.ax.get_ylim()
            x_center = (xlim[0] + xlim[1]) / 2
            y_center = (ylim[0] + ylim[1]) / 2
            x_range = (xlim[1] - xlim[0]) / self.zoom_level
            y_range = (ylim[1] - ylim[0]) / self.zoom_level
            self.ax.set_xlim(x_center - x_range/2, x_center + x_range/2)
            self.ax.set_ylim(y_center - y_range/2, y_center + y_range/2)
        
        self.canvas.draw()
    
    # =========================================================================
    # CONTROLES DE NAVEGAÇÃO E ZOOM
    # =========================================================================
    def mudar_fatia(self, value):
        self.current_slice_index = int(float(value))
        self.exibir_imagem()
    
    def zoom_in(self):
        self.zoom_level *= 1.2
        self.lbl_zoom.config(text=f"Zoom: {int(self.zoom_level*100)}%")
        self.exibir_imagem()
    
    def zoom_out(self):
        self.zoom_level /= 1.2
        if self.zoom_level < 0.5:
            self.zoom_level = 0.5
        self.lbl_zoom.config(text=f"Zoom: {int(self.zoom_level*100)}%")
        self.exibir_imagem()
    
    def zoom_reset(self):
        self.zoom_level = 1.0
        self.lbl_zoom.config(text=f"Zoom: {int(self.zoom_level*100)}%")
        self.exibir_imagem()
    
    # =========================================================================
    # FUNÇÕES DE PROCESSAMENTO
    # =========================================================================
    def preparar_dados(self):
        """Prepara dados com split 80/20"""
        try:
            messagebox.showinfo("Preparando Dados", "Carregando e preparando dados...\nIsso pode levar alguns minutos.")
            self.train_data, self.val_data, self.test_data, self.valid_data_df, self.test_patients = load_and_prepare_data()
            messagebox.showinfo("Sucesso", f"Dados preparados com sucesso!\n\n"
                              f"Treino: {len(self.train_data[0])} imagens\n"
                              f"Validação: {len(self.val_data[0])} imagens\n"
                              f"Teste: {len(self.test_data[0])} imagens")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao preparar dados:\n{str(e)}")
    
    def extrair_caracteristicas(self):
        """Extrai características GLCM das imagens"""
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        try:
            messagebox.showinfo("Extraindo", "Extraindo características...\nIsso pode levar alguns minutos.")
            self.x_train_features = extract_features(self.train_data[0])
            self.x_val_features = extract_features(self.val_data[0])
            self.x_test_features = extract_features(self.test_data[0])
            messagebox.showinfo("Sucesso", "Características extraídas com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao extrair características:\n{str(e)}")
    
    # =========================================================================
    # SVM - CLASSIFICADOR RASO
    # =========================================================================
    def treinar_svm(self):
        """Treina SVM"""
        if self.x_train_features is None:
            messagebox.showwarning("Aviso", "Extraia as características primeiro!")
            return
        
        try:
            messagebox.showinfo("Treinando", "Treinando SVM...")
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.x_train_features)
            self.svm_model = SVC(kernel='rbf', C=1.0, gamma='scale', random_state=42)
            self.svm_model.fit(X_train_scaled, self.train_data[1])
            self.svm_scaler = scaler
            messagebox.showinfo("Sucesso", "SVM treinado com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar SVM:\n{str(e)}")
    
    def avaliar_svm(self):
        """Avalia SVM no conjunto de teste"""
        if self.svm_model is None:
            messagebox.showwarning("Aviso", "Treine o SVM primeiro!")
            return
        
        try:
            X_test_scaled = self.svm_scaler.transform(self.x_test_features)
            y_pred = self.svm_model.predict(X_test_scaled)
            acc, sens, spec, cm = evaluate_classifier(self.test_data[1], y_pred, "SVM")
            
            messagebox.showinfo("Resultados SVM", 
                              f"Acurácia: {acc:.4f}\n"
                              f"Sensibilidade: {sens:.4f}\n"
                              f"Especificidade: {spec:.4f}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao avaliar SVM:\n{str(e)}")
    
    def matriz_confusao_svm(self):
        """Mostra matriz de confusão do SVM"""
        if self.svm_model is None:
            messagebox.showwarning("Aviso", "Treine o SVM primeiro!")
            return
        
        try:
            X_test_scaled = self.svm_scaler.transform(self.x_test_features)
            y_pred = self.svm_model.predict(X_test_scaled)
            cm = confusion_matrix(self.test_data[1], y_pred)
            
            plt.figure(figsize=(6, 5))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
            plt.title('Matriz de Confusão - SVM')
            plt.ylabel('Real')
            plt.xlabel('Predito')
            plt.show()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
    
    # =========================================================================
    # XGBOOST - REGRESSOR RASO
    # =========================================================================
    def treinar_xgboost(self):
        """Treina XGBoost para regressão de idade"""
        if self.x_train_features is None:
            messagebox.showwarning("Aviso", "Extraia as características primeiro!")
            return
        
        try:
            messagebox.showinfo("Treinando", "Treinando XGBoost...")
            self.xgboost_model = xgb.XGBRegressor(n_estimators=100, max_depth=5, learning_rate=0.1, random_state=42)
            self.xgboost_model.fit(self.x_train_features, self.train_data[2])
            messagebox.showinfo("Sucesso", "XGBoost treinado com sucesso!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar XGBoost:\n{str(e)}")
    
    def avaliar_xgboost(self):
        """Avalia XGBoost"""
        if self.xgboost_model is None:
            messagebox.showwarning("Aviso", "Treine o XGBoost primeiro!")
            return
        
        try:
            y_pred = self.xgboost_model.predict(self.x_test_features)
            mae = mean_absolute_error(self.test_data[2], y_pred)
            messagebox.showinfo("Resultados XGBoost", f"MAE: {mae:.2f} anos")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
    
    def analise_temporal(self):
        """Análise temporal - visitas posteriores têm idades maiores?"""
        if self.xgboost_model is None:
            messagebox.showwarning("Aviso", "Treine o XGBoost primeiro!")
            return
        
        try:
            y_pred = self.xgboost_model.predict(self.x_test_features)
            test_data_df = self.valid_data_df[self.valid_data_df['Subject ID'].isin(self.test_patients['Subject ID'])]
            test_data_df = test_data_df.copy()
            test_data_df['predicted_age'] = y_pred[:len(test_data_df)]
            
            consistent_increase = 0
            total_patients = 0
            
            for subject_id, group in test_data_df.sort_values(['Subject ID', 'Visit']).groupby('Subject ID'):
                if len(group) > 1:
                    total_patients += 1
                    predicted_ages = group['predicted_age'].values
                    if np.all(np.diff(predicted_ages) >= 0):
                        consistent_increase += 1
            
            ratio = (consistent_increase / total_patients * 100) if total_patients > 0 else 0
            messagebox.showinfo("Análise Temporal", 
                              f"Pacientes com múltiplas visitas: {total_patients}\n"
                              f"Com idades crescentes: {consistent_increase}\n"
                              f"Percentual: {ratio:.1f}%")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
    
    # =========================================================================
    # DENSENET - MODELO PROFUNDO
    # =========================================================================
    def treinar_densenet_classif(self):
        """Treina DenseNet para classificação"""
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        messagebox.showinfo("DenseNet", "Treinamento de DenseNet para classificação\n"
                          "será implementado aqui com fine-tuning.")
    
    def treinar_densenet_regress(self):
        """Treina DenseNet para regressão"""
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        messagebox.showinfo("DenseNet", "Treinamento de DenseNet para regressão\n"
                          "será implementado aqui com fine-tuning.")
    
    def avaliar_densenet_classif(self):
        """Avalia DenseNet classificação"""
        messagebox.showinfo("DenseNet", "Avaliação de DenseNet Classificação")
    
    def avaliar_densenet_regress(self):
        """Avalia DenseNet regressão"""
        messagebox.showinfo("DenseNet", "Avaliação de DenseNet Regressão")
    
    # =========================================================================
    # ACESSIBILIDADE
    # =========================================================================
    def atualizar_fontes(self):
        """Atualiza o tamanho de todas as fontes"""
        self.base_font = ("Arial", self.font_size)
        self.style.configure("TButton", font=self.base_font)
        self.style.configure("TLabel", font=self.base_font)
        
        # Atualiza labels
        for widget in [self.lbl_arquivo, self.lbl_formato, self.lbl_dimensoes, 
                      self.lbl_slice, self.lbl_zoom]:
            widget.config(font=self.base_font)
    
    def aumentar_fonte(self):
        """Aumenta o tamanho da fonte"""
        self.font_size += 2
        self.atualizar_fontes()
        messagebox.showinfo("Acessibilidade", f"Fonte aumentada para {self.font_size}pt")
    
    def diminuir_fonte(self):
        """Diminui o tamanho da fonte"""
        if self.font_size > 8:
            self.font_size -= 2
            self.atualizar_fontes()
            messagebox.showinfo("Acessibilidade", f"Fonte reduzida para {self.font_size}pt")

# =============================================================================
# MAIN
# =============================================================================
if __name__ == "__main__":
    root = tk.Tk()
    app = AlzheimerAnalysisGUI(root)
    root.mainloop()
