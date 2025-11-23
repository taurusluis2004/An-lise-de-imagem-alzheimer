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
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, mean_absolute_error, balanced_accuracy_score, f1_score
from sklearn.decomposition import PCA
from sklearn.utils.class_weight import compute_class_weight
import xgboost as xgb
import nibabel as nib
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from keras.models import Model, Sequential
from keras.layers import Input, Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D, GlobalAveragePooling2D, RandomFlip, RandomRotation, RandomZoom, RandomTranslation
from keras.applications import DenseNet121
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping, ReduceLROnPlateau, Callback

# =============================================================================
# CALLBACK PARA MÉTRICAS PERSONALIZADAS (DenseNet)
# =============================================================================
class MetricsCallbackDenseNet(Callback):
    def __init__(self, x_val, y_val):
        super().__init__()
        self.x_val = x_val
        self.y_val = y_val
        self.history_sensitivity = []
        self.history_specificity = []

    def on_epoch_end(self, epoch, logs=None):
        # Previsões no conjunto de validação
        y_pred_prob = self.model.predict(self.x_val, verbose=0)
        y_pred = (y_pred_prob.flatten() >= 0.5).astype(int)
        cm = confusion_matrix(self.y_val, y_pred)
        if cm.shape == (2,2):
            tn, fp, fn, tp = cm.ravel()
            sens = tp / (tp + fn + 1e-8)
            spec = tn / (tn + fp + 1e-8)
        else:
            sens = 0.0
            spec = 0.0
        self.history_sensitivity.append(sens)
        self.history_specificity.append(spec)
        print(f"Epoch {epoch+1}: Sensibilidade={sens:.4f} | Especificidade={spec:.4f}")

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
    """Extracts robust texture features from images (Windows-safe logging)."""
    features = []
    total = len(images)
    print(f"Extraindo features de {total} imagens...")

    for idx, img in enumerate(images):
        if idx % 50 == 0:
            print(f"  Progresso: {idx}/{total}")

        try:
            # Garantir array float sem NaN/Inf
            img = np.nan_to_num(img, nan=0.0, posinf=1.0, neginf=0.0)

            # Normalizar para 0-255 de forma estável
            vmin = float(np.min(img))
            vmax = float(np.max(img))
            if vmax > vmin:
                img_norm = (img - vmin) / (vmax - vmin)
            else:
                img_norm = np.zeros_like(img)
            img_8bit = np.clip((img_norm * 255).round(), 0, 255).astype(np.uint8)

            feature_vec = []

            # GLCM com múltiplos ângulos (mais informação de textura)
            angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
            glcm = graycomatrix(
                img_8bit,
                distances=[1],
                angles=angles,
                levels=256,
                symmetric=True,
                normed=True,
            )

            # 6 propriedades GLCM para cada ângulo = 24 features
            for prop in [
                'contrast',
                'dissimilarity',
                'homogeneity',
                'energy',
                'correlation',
                'ASM',
            ]:
                vals = graycoprops(glcm, prop)[0, :]
                feature_vec.extend(vals.tolist())

            # Features estatísticas adicionais (8 features)
            feature_vec.extend(
                [
                    float(np.mean(img)),
                    float(np.std(img)),
                    float(np.median(img)),
                    float(np.min(img)),
                    float(np.max(img)),
                    float(np.max(img) - np.min(img)),  # Range
                    float(np.percentile(img, 25)),  # Q1
                    float(np.percentile(img, 75)),  # Q3
                ]
            )

            features.append(feature_vec)
        except Exception as ex:
            # Em caso de falha em uma imagem, registra e continua
            print(f"  Aviso: falha ao extrair da imagem {idx}: {ex}")
            # Preenche com zeros para manter alinhamento
            if features:
                features.append([0.0] * len(features[0]))
            else:
                # 24 GLCM + 8 estatísticas = 32
                features.append([0.0] * 32)

    if features and len(features[0]) > 0:
        print(f"Total de {len(features[0])} features por imagem")
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
        menu_densenet.add_command(label="Análise Temporal", command=self.analise_temporal_densenet)
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
            base_dir = os.path.dirname(os.path.abspath(__file__))
            cache_path = os.path.join(base_dir, 'features_cache.npz')

            if os.path.exists(cache_path):
                usar_cache = messagebox.askyesno(
                    "Cache de Features",
                    "Encontrado arquivo de cache (features_cache.npz).\nDeseja reutilizar em vez de recalcular?"
                )
                if usar_cache:
                    data = np.load(cache_path, allow_pickle=True)
                    self.x_train_features = data['x_train']
                    self.x_val_features = data['x_val']
                    self.x_test_features = data['x_test']
                    messagebox.showinfo("Sucesso", "Cache carregado com sucesso!")
                    return

            messagebox.showinfo("Extraindo", "Calculando features (GLCM + estatísticas)...")
            self.x_train_features = extract_features(self.train_data[0])
            self.x_val_features = extract_features(self.val_data[0])
            self.x_test_features = extract_features(self.test_data[0])

            np.savez_compressed(
                cache_path,
                x_train=self.x_train_features,
                x_val=self.x_val_features,
                x_test=self.x_test_features
            )
            messagebox.showinfo("Sucesso", "Features extraídas e cache salvo!")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao extrair características:\n{str(e)}")

    # =========================================================================
    # HELPER: TRANSFORM FEATURES (Scaler + PCA)
    # =========================================================================
    def transform_features(self, X):
        """Aplica scaler e PCA (se disponível) nas features."""
        if not hasattr(self, 'svm_scaler'):
            raise ValueError("Scaler não encontrado. Treine o SVM primeiro.")
        X_scaled = self.svm_scaler.transform(X)
        if hasattr(self, 'svm_pca') and self.svm_pca is not None:
            X_scaled = self.svm_pca.transform(X_scaled)
        return X_scaled
    
    # =========================================================================
    # SVM - CLASSIFICADOR RASO
    # =========================================================================
    def treinar_svm(self):
        """Treina SVM com PCA adaptativo e busca de hiperparâmetros.

        Objetivo: melhorar Acurácia / Sensibilidade / Especificidade >= 0.75.
        Pipeline: StandardScaler -> PCA (variância 0.98 máx 40 comps) -> GridSearchCV.
        """
        if self.x_train_features is None:
            messagebox.showwarning("Aviso", "Extraia as características primeiro!")
            return
        
        try:
            messagebox.showinfo("Treinando", "Treinando SVM (Scaler + PCA + GridSearchCV)...")

            # 1) Normalização
            scaler = StandardScaler()
            X_train_scaled = scaler.fit_transform(self.x_train_features)

            # 2) PCA adaptativo: mantém até 98% da variância, limite de 40 componentes
            pca_full = PCA(svd_solver='full')
            pca_full.fit(X_train_scaled)
            cum_var = np.cumsum(pca_full.explained_variance_ratio_)
            n_comp = int(np.searchsorted(cum_var, 0.98) + 1)
            n_comp = min(n_comp, 40, X_train_scaled.shape[1], X_train_scaled.shape[0]-1)
            pca = PCA(n_components=n_comp, svd_solver='full')
            X_train_pca = pca.fit_transform(X_train_scaled)

            # 3) Grid pequeno para não demorar, com validação estratificada
            param_grid = [
                {'kernel': ['rbf'], 'C': [1, 5, 10, 20], 'gamma': ['scale', 0.01, 0.1]},
                {'kernel': ['linear'], 'C': [0.5, 1, 5, 10]}
            ]
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            grid = GridSearchCV(
                SVC(class_weight='balanced', random_state=42),
                param_grid=param_grid,
                scoring='balanced_accuracy',
                cv=cv,
                n_jobs=-1,
                refit=True,
                verbose=0
            )
            grid.fit(X_train_pca, self.train_data[1])

            self.svm_model = grid.best_estimator_
            self.svm_scaler = scaler
            self.svm_pca = pca

            train_bal_acc = balanced_accuracy_score(self.train_data[1], self.svm_model.predict(X_train_pca))
            msg = (
                f"SVM treinado!\n\n"
                f"Melhores hiperparâmetros: {grid.best_params_}\n"
                f"Score CV (Balanced Acc): {grid.best_score_:.3f}\n"
                f"Balanced Acc Treino: {train_bal_acc:.3f}\n"
                f"Componentes PCA: {X_train_pca.shape[1]} (Var ≈ {cum_var[n_comp-1]*100:.1f}%)"
            )
            messagebox.showinfo("Sucesso", msg)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar SVM:\n{str(e)}")
    
    def avaliar_svm(self):
        """Avalia SVM no conjunto de teste"""
        if self.svm_model is None:
            messagebox.showwarning("Aviso", "Treine o SVM primeiro!")
            return
        
        try:
            y_pred = self.svm_model.predict(self.transform_features(self.x_test_features))
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
            X_test_trans = self.transform_features(self.x_test_features)
            y_pred = self.svm_model.predict(X_test_trans)
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
        """Avalia XGBoost no conjunto de TESTE"""
        if self.xgboost_model is None:
            messagebox.showwarning("Aviso", "Treine o XGBoost primeiro!")
            return
        
        try:
            y_pred = self.xgboost_model.predict(self.x_test_features)
            y_true = self.test_data[2]
            
            mae = mean_absolute_error(y_true, y_pred)
            
            from sklearn.metrics import r2_score, mean_squared_error
            r2 = r2_score(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            
            # Avaliar qualidade da predição
            qualidade = "Excelente" if mae < 5 else "Bom" if mae < 10 else "Requer melhorias"
            
            result_text = (
                f"=== RESULTADOS XGBoost (TESTE) ===\n\n"
                f"MAE: {mae:.2f} anos\n"
                f"RMSE: {rmse:.2f} anos\n"
                f"R² Score: {r2:.4f}\n\n"
                f"Qualidade: {qualidade}\n\n"
                f"Interpretação:\n"
                f"• MAE < 5 anos: Excelente\n"
                f"• MAE 5-10 anos: Bom\n"
                f"• MAE > 10 anos: Requer melhorias\n\n"
                f"Resposta: As características GLCM {'SÃO' if mae < 10 else 'NÃO SÃO'} suficientes\n"
                f"para obter uma boa predição de idade."
            )
            
            messagebox.showinfo("Resultados XGBoost", result_text)
            
            # Plotar gráficos comparativos
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            # Scatter plot
            ax1.scatter(y_true, y_pred, alpha=0.6, s=50, edgecolors='black')
            ax1.plot([y_true.min(), y_true.max()], 
                    [y_true.min(), y_true.max()], 
                    'r--', lw=2, label='Predição Perfeita')
            ax1.set_xlabel('Idade Real (anos)', fontsize=12)
            ax1.set_ylabel('Idade Predita (anos)', fontsize=12)
            ax1.set_title(f'XGBoost - Predição de Idade\nMAE: {mae:.2f} anos | R²: {r2:.4f}', 
                         fontsize=14, fontweight='bold')
            ax1.legend()
            ax1.grid(True, alpha=0.3)
            
            # Histograma dos erros
            erros = y_pred - y_true
            ax2.hist(erros, bins=20, edgecolor='black', alpha=0.7, color='skyblue')
            ax2.axvline(x=0, color='r', linestyle='--', linewidth=2, label='Erro Zero')
            ax2.set_xlabel('Erro de Predição (anos)', fontsize=12)
            ax2.set_ylabel('Frequência', fontsize=12)
            ax2.set_title(f'Distribuição dos Erros\nMédia: {erros.mean():.2f} anos | Desvio: {erros.std():.2f}', 
                         fontsize=14, fontweight='bold')
            ax2.legend()
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def analise_temporal(self):
        """Análise temporal COMPLETA - visitas posteriores têm idades maiores?"""
        if self.xgboost_model is None:
            messagebox.showwarning("Aviso", "Treine o XGBoost primeiro!")
            return
        
        try:
            y_pred = self.xgboost_model.predict(self.x_test_features)
            test_data_df = self.valid_data_df[self.valid_data_df['Subject ID'].isin(self.test_patients['Subject ID'])].copy()
            
            # Adicionar predições
            if len(y_pred) != len(test_data_df):
                messagebox.showwarning("Aviso", "Número de predições não corresponde aos dados de teste")
                return
            
            test_data_df['predicted_age'] = y_pred
            
            consistent_increase = 0
            total_patients = 0
            exemplos_bons = []
            exemplos_ruins = []
            diferencas_medias = []
            
            # Analisar cada paciente com múltiplas visitas
            for subject_id, group in test_data_df.sort_values(['Subject ID', 'Visit']).groupby('Subject ID'):
                if len(group) > 1:
                    total_patients += 1
                    predicted_ages = group['predicted_age'].values
                    real_ages = group['Age'].values
                    
                    # Verificar se idades preditas são crescentes
                    diffs_pred = np.diff(predicted_ages)
                    diffs_real = np.diff(real_ages)
                    
                    if np.all(diffs_pred >= 0):
                        consistent_increase += 1
                        if len(exemplos_bons) < 3:
                            exemplos_bons.append((subject_id, predicted_ages, real_ages))
                    else:
                        if len(exemplos_ruins) < 3:
                            exemplos_ruins.append((subject_id, predicted_ages, real_ages))
                    
                    diferencas_medias.append(np.mean(diffs_pred))
            
            ratio = (consistent_increase / total_patients * 100) if total_patients > 0 else 0
            
            # Criar relatório detalhado
            result_text = (
                f"=== ANÁLISE TEMPORAL (XGBoost) ===\n\n"
                f"Pacientes com múltiplas visitas: {total_patients}\n"
                f"Com idades CRESCENTES: {consistent_increase}\n"
                f"Percentual de acerto: {ratio:.1f}%\n\n"
                f"Diferença média entre visitas: {np.mean(diferencas_medias):.2f} anos\n\n"
                f"RESPOSTA: {'SIM' if ratio >= 70 else 'NÃO'}, {ratio:.1f}% dos exames\n"
                f"em visitas posteriores resultaram em idades\n"
                f"{'maiores ou iguais' if ratio >= 70 else 'NEM SEMPRE maiores'}.\n\n"
            )
            
            # Adicionar exemplos
            if exemplos_bons:
                result_text += "Exemplos de SUCESSO:\n"
                for subj_id, pred_ages, real_ages in exemplos_bons[:2]:
                    result_text += f"  {subj_id}: {pred_ages[0]:.1f}→{pred_ages[-1]:.1f} anos (Real: {real_ages[0]:.1f}→{real_ages[-1]:.1f})\n"
            
            if exemplos_ruins:
                result_text += "\nExemplos de FALHA:\n"
                for subj_id, pred_ages, real_ages in exemplos_ruins[:2]:
                    result_text += f"  {subj_id}: {pred_ages[0]:.1f}→{pred_ages[-1]:.1f} anos (Real: {real_ages[0]:.1f}→{real_ages[-1]:.1f})\n"
            
            messagebox.showinfo("Análise Temporal - XGBoost", result_text)
            
            # Plotar gráfico de progressão temporal
            if exemplos_bons:
                fig, axes = plt.subplots(1, min(3, len(exemplos_bons)), figsize=(15, 4))
                if len(exemplos_bons) == 1:
                    axes = [axes]
                
                for idx, (subj_id, pred_ages, real_ages) in enumerate(exemplos_bons[:3]):
                    ax = axes[idx] if len(exemplos_bons) > 1 else axes[0]
                    visitas = range(1, len(pred_ages) + 1)
                    
                    ax.plot(visitas, real_ages, 'g-o', label='Idade Real', linewidth=2, markersize=8)
                    ax.plot(visitas, pred_ages, 'b--s', label='Idade Predita', linewidth=2, markersize=8)
                    ax.set_xlabel('Visita', fontsize=11)
                    ax.set_ylabel('Idade (anos)', fontsize=11)
                    ax.set_title(f'Paciente {subj_id}\n(Exemplo de Sucesso)', fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.suptitle('Progressão Temporal - Exemplos de Predições Corretas', 
                           fontsize=14, fontweight='bold', y=1.02)
                plt.show()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    # =========================================================================
    # DENSENET - MODELO PROFUNDO
    # =========================================================================
    def treinar_densenet_classif(self):
        """Treina DenseNet121 com augmentation + fine-tuning progressivo.

        Objetivo: melhorar métricas (acurácia/sensibilidade/especificidade >= 0.75).
        Modo rápido: menos épocas & subset.
        """
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        try:
            # Perguntar modo de treinamento
            resposta = messagebox.askyesno(
                "Modo de Treinamento",
                "Deseja usar modo RÁPIDO (otimizado para CPU)?\n\n"
                "SIM = Rápido (10 épocas, batch 32, menos amostras)\n"
                "NÃO = Completo (30 épocas, batch 16, todos dados)\n\n"
                "Ambos usam DenseNet121 com pesos ImageNet."
            )
            
            modo_rapido = resposta
            
            messagebox.showinfo("Treinando", 
                              f"Treinando DenseNet121 em modo {'RÁPIDO' if modo_rapido else 'COMPLETO'}...\n"
                              "Usando pesos pré-treinados do ImageNet.\n"
                              "Aguarde alguns minutos.")
            
            # Preparar dados (repete canais para compatibilidade ImageNet)
            x_train = np.expand_dims(self.train_data[0], axis=-1)
            x_train = np.repeat(x_train, 3, axis=-1)
            x_val = np.expand_dims(self.val_data[0], axis=-1)
            x_val = np.repeat(x_val, 3, axis=-1)
            
            y_train = self.train_data[1]
            y_val = self.val_data[1]
            
            # OTIMIZAÇÃO 1: Reduzir dados em modo rápido
            if modo_rapido and len(x_train) > 150:
                indices = np.random.choice(len(x_train), 150, replace=False)
                x_train = x_train[indices]
                y_train = y_train[indices]
                print(f"Modo rápido: usando {len(x_train)} amostras de treino")
            
            print(f"Treinamento - Shape: {x_train.shape}, Labels: {y_train.shape}")
            
            # Modelo base DenseNet121 com ImageNet
            base_model = DenseNet121(
                weights='imagenet',
                include_top=False,
                input_shape=(128, 128, 3)
            )
            
            print("Usando DenseNet121 com pesos ImageNet")
            
            # Fase 1: congelar tudo exceto últimas 20 camadas para estabilizar
            for layer in base_model.layers[:-20]:
                layer.trainable = False
            for layer in base_model.layers[-20:]:
                layer.trainable = True
            print("Fase 1: Fine-tuning últimas 20 camadas")
            
            # Classificador (GAP ajuda generalização com menos parâmetros)
            # Augmentation via camadas (sem ImageDataGenerator para compatibilidade Keras 3)
            augmentation = Sequential([
                RandomFlip('horizontal'),
                RandomRotation(0.05),
                RandomZoom(0.1),
                RandomTranslation(0.05, 0.05),
            ], name='augmentation')

            inputs = Input(shape=(128,128,3))
            x = augmentation(inputs)
            x = base_model(x, training=False)
            x = GlobalAveragePooling2D()(x)
            x = BatchNormalization()(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.4)(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            predictions = Dense(1, activation='sigmoid')(x)
            
            self.densenet_classif = Model(inputs=inputs, outputs=predictions)
            
            self.densenet_classif.compile(
                optimizer=Adam(learning_rate=1e-4),
                loss='binary_crossentropy',
                metrics=['accuracy']
            )
            
            # Parâmetros de treinamento
            epochs_phase1 = 12 if modo_rapido else 25
            epochs_phase2 = 5 if modo_rapido else 20
            batch_size = 16 if modo_rapido else 12
            patience = 4 if modo_rapido else 10
            
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=patience,
                restore_best_weights=True, 
                verbose=1
            )
            
            # Class weights para lidar com desbalanceamento
            classes = np.unique(y_train)
            class_w = compute_class_weight(class_weight='balanced', classes=classes, y=y_train)
            class_weight_dict = {int(c): float(w) for c, w in zip(classes, class_w)}

            metrics_cb = MetricsCallbackDenseNet(x_val, y_val)

            print(f"Fase 1: treinamento ({epochs_phase1} épocas, batch={batch_size})...")
            hist1 = self.densenet_classif.fit(
                x_train, y_train,
                validation_data=(x_val, y_val),
                epochs=epochs_phase1,
                callbacks=[early_stop, metrics_cb],
                class_weight=class_weight_dict,
                verbose=1
            )

            # Fase 2: ampliar fine-tuning descongelando mais camadas (se modo completo)
            if not modo_rapido:
                print("Fase 2: descongelando últimas 50 camadas para fine-tuning profundo...")
                for layer in base_model.layers[:-50]:
                    layer.trainable = False
                for layer in base_model.layers[-50:]:
                    layer.trainable = True
                # Reduzir learning rate para evitar destruição dos pesos
                self.densenet_classif.compile(
                    optimizer=Adam(learning_rate=5e-5),
                    loss='binary_crossentropy',
                    metrics=['accuracy']
                )
                early_stop_phase2 = EarlyStopping(monitor='val_loss', patience=patience, restore_best_weights=True, verbose=1)
                print(f"Fase 2: treinamento adicional ({epochs_phase2} épocas)...")
                hist2 = self.densenet_classif.fit(
                    x_train, y_train,
                    validation_data=(x_val, y_val),
                    epochs=epochs_phase2,
                    callbacks=[early_stop_phase2, metrics_cb],
                    class_weight=class_weight_dict,
                    verbose=1
                )
            else:
                hist2 = None

            # Combinar histórico
            self.densenet_history = {
                'accuracy': hist1.history.get('accuracy', []) + ([] if hist2 is None else hist2.history.get('accuracy', [])),
                'val_accuracy': hist1.history.get('val_accuracy', []) + ([] if hist2 is None else hist2.history.get('val_accuracy', [])),
                'loss': hist1.history.get('loss', []) + ([] if hist2 is None else hist2.history.get('loss', [])),
                'val_loss': hist1.history.get('val_loss', []) + ([] if hist2 is None else hist2.history.get('val_loss', [])),
                'val_sensitivity': metrics_cb.history_sensitivity,
                'val_specificity': metrics_cb.history_specificity
            }
            
            final_acc = self.densenet_history['val_accuracy'][-1]
            messagebox.showinfo("Sucesso", 
                              f"DenseNet121 treinado com sucesso!\n\n"
                              f"Modelo: DenseNet121 (ImageNet)\n"
                              f"Épocas completadas: {len(self.densenet_history['loss'])}\n"
                              f"Acurácia final: {final_acc:.4f}\n\n"
                              f"Modo: {'RÁPIDO (CPU)' if modo_rapido else 'COMPLETO'}")
            
            self.plot_densenet_learning_curves()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar DenseNet:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def treinar_densenet_regress(self):
        """Treina DenseNet121 OTIMIZADO para regressão de idade"""
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        try:
            resposta = messagebox.askyesno(
                "Modo de Treinamento",
                "Usar modo RÁPIDO (otimizado para CPU)?\n\n"
                "Ambos usam DenseNet121 com pesos ImageNet."
            )
            
            modo_rapido = resposta
            
            messagebox.showinfo("Treinando", 
                              f"Treinando DenseNet121 para regressão...\n"
                              f"Modo: {'RÁPIDO' if modo_rapido else 'COMPLETO'}")
            
            x_train = np.expand_dims(self.train_data[0], axis=-1)
            x_train = np.repeat(x_train, 3, axis=-1)
            x_val = np.expand_dims(self.val_data[0], axis=-1)
            x_val = np.repeat(x_val, 3, axis=-1)
            
            y_train_age = self.train_data[2]
            y_val_age = self.val_data[2]
            
            # Reduzir dados em modo rápido
            if modo_rapido and len(x_train) > 150:
                indices = np.random.choice(len(x_train), 150, replace=False)
                x_train = x_train[indices]
                y_train_age = y_train_age[indices]
                print(f"Modo rápido: {len(x_train)} amostras")
            
            # Sempre usar DenseNet121
            base_model = DenseNet121(
                weights='imagenet', 
                include_top=False, 
                input_shape=(128, 128, 3)
            )
            
            print("Usando DenseNet121 com pesos ImageNet")
            
            # Congelar todas as camadas
            for layer in base_model.layers:
                layer.trainable = False
            
            print(f"Todas as {len(base_model.layers)} camadas congeladas")
            
            x = base_model.output
            x = BatchNormalization()(x)
            x = Flatten()(x)
            if modo_rapido:
                x = Dense(128, activation='relu')(x)
                x = Dropout(0.3)(x)
            else:
                x = Dense(256, activation='relu')(x)
                x = Dropout(0.5)(x)
                x = Dense(128, activation='relu')(x)
                x = Dropout(0.3)(x)
            predictions = Dense(1, activation='linear')(x)
            
            self.densenet_regress = Model(inputs=base_model.input, outputs=predictions)
            
            self.densenet_regress.compile(
                optimizer=Adam(learning_rate=0.001 if modo_rapido else 0.0001),
                loss='mse',
                metrics=['mae']
            )
            
            epochs = 10 if modo_rapido else 30
            batch_size = 32 if modo_rapido else 16
            patience = 3 if modo_rapido else 10
            
            early_stop = EarlyStopping(
                monitor='val_loss', 
                patience=patience, 
                restore_best_weights=True, 
                verbose=1
            )
            
            print(f"Treinando ({epochs} épocas, batch={batch_size})...")
            
            self.densenet_regress_history = self.densenet_regress.fit(
                x_train, y_train_age,
                validation_data=(x_val, y_val_age),
                epochs=epochs,
                batch_size=batch_size,
                callbacks=[early_stop],
                verbose=1
            )
            
            final_mae = self.densenet_regress_history.history['val_mae'][-1]
            messagebox.showinfo("Sucesso", 
                              f"DenseNet121 (regressão) treinado!\n\n"
                              f"Modelo: DenseNet121 (ImageNet)\n"
                              f"MAE final: {final_mae:.2f} anos\n"
                              f"Modo: {'RÁPIDO' if modo_rapido else 'COMPLETO'}")
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def avaliar_densenet_classif(self):
        """Avalia DenseNet classificação no TESTE"""
        if self.densenet_classif is None:
            messagebox.showwarning("Aviso", "Treine o DenseNet primeiro!")
            return
        
        try:
            x_test = np.expand_dims(self.test_data[0], axis=-1)
            x_test = np.repeat(x_test, 3, axis=-1)
            y_test = self.test_data[1]
            
            y_pred_prob = self.densenet_classif.predict(x_test, verbose=0)
            y_pred = (y_pred_prob > 0.5).astype(int).flatten()
            
            acc, sens, spec, cm = evaluate_classifier(y_test, y_pred, "DenseNet121")
            
            messagebox.showinfo("Resultados DenseNet", 
                              f"=== TESTE ===\n\n"
                              f"Acurácia: {acc:.4f} ({acc*100:.2f}%)\n"
                              f"Sensibilidade: {sens:.4f}\n"
                              f"Especificidade: {spec:.4f}")
            
            plt.figure(figsize=(7, 6))
            sns.heatmap(cm, annot=True, fmt='d', cmap='Greens', cbar=True)
            plt.title(f'Matriz de Confusão - DenseNet121\nAcurácia: {acc:.4f}')
            plt.ylabel('Classe Real')
            plt.xlabel('Classe Predita')
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def avaliar_densenet_regress(self):
        """Avalia DenseNet regressão no TESTE"""
        if self.densenet_regress is None:
            messagebox.showwarning("Aviso", "Treine o DenseNet (regressão) primeiro!")
            return
        
        try:
            x_test = np.expand_dims(self.test_data[0], axis=-1)
            x_test = np.repeat(x_test, 3, axis=-1)
            y_test_age = self.test_data[2]
            
            y_pred_age = self.densenet_regress.predict(x_test, verbose=0).flatten()
            mae = mean_absolute_error(y_test_age, y_pred_age)
            
            from sklearn.metrics import r2_score
            r2 = r2_score(y_test_age, y_pred_age)
            
            messagebox.showinfo("Resultados DenseNet (Regressão)", 
                              f"=== TESTE ===\n\n"
                              f"MAE: {mae:.2f} anos\n"
                              f"R² Score: {r2:.4f}\n\n"
                              f"Interpretação:\n"
                              f"• < 5 anos: Excelente\n"
                              f"• 5-10 anos: Bom\n"
                              f"• > 10 anos: Requer melhorias")
            
            fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))
            
            ax1.scatter(y_test_age, y_pred_age, alpha=0.6, s=50)
            ax1.plot([y_test_age.min(), y_test_age.max()], 
                    [y_test_age.min(), y_test_age.max()], 'r--', lw=2)
            ax1.set_xlabel('Idade Real')
            ax1.set_ylabel('Idade Predita')
            ax1.set_title(f'DenseNet - Regressão\nMAE: {mae:.2f} anos | R²: {r2:.4f}')
            ax1.grid(True, alpha=0.3)
            
            erros = y_pred_age - y_test_age
            ax2.hist(erros, bins=20, edgecolor='black', alpha=0.7)
            ax2.axvline(x=0, color='r', linestyle='--', linewidth=2)
            ax2.set_xlabel('Erro (anos)')
            ax2.set_ylabel('Frequência')
            ax2.set_title(f'Distribuição dos Erros\nMédia: {erros.mean():.2f} anos')
            ax2.grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
    def plot_densenet_learning_curves(self):
        """Plota curvas de aprendizado: Acurácia, Loss, Sensibilidade e Especificidade."""
        if not hasattr(self, 'densenet_history'):
            messagebox.showwarning("Aviso", "Nenhum histórico de treinamento!")
            return
        
        fig, axes = plt.subplots(2, 2, figsize=(15, 10))
        epochs = range(1, len(self.densenet_history['accuracy']) + 1)

        # Acurácia
        axes[0,0].plot(epochs, self.densenet_history['accuracy'], 'b-o', label='Treino', linewidth=2)
        axes[0,0].plot(epochs, self.densenet_history['val_accuracy'], 'r-s', label='Validação', linewidth=2)
        axes[0,0].set_title('Acurácia', fontsize=13, fontweight='bold')
        axes[0,0].set_xlabel('Época')
        axes[0,0].set_ylabel('Acurácia')
        axes[0,0].legend()
        axes[0,0].grid(True, alpha=0.3)

        # Loss
        axes[0,1].plot(epochs, self.densenet_history['loss'], 'b-o', label='Treino', linewidth=2)
        axes[0,1].plot(epochs, self.densenet_history['val_loss'], 'r-s', label='Validação', linewidth=2)
        axes[0,1].set_title('Loss', fontsize=13, fontweight='bold')
        axes[0,1].set_xlabel('Época')
        axes[0,1].set_ylabel('Loss')
        axes[0,1].legend()
        axes[0,1].grid(True, alpha=0.3)

        # Sensibilidade
        if 'val_sensitivity' in self.densenet_history:
            axes[1,0].plot(epochs, self.densenet_history['val_sensitivity'], 'g-o', label='Sensibilidade', linewidth=2)
            axes[1,0].set_title('Sensibilidade (Validação)', fontsize=13, fontweight='bold')
            axes[1,0].set_xlabel('Época')
            axes[1,0].set_ylabel('Sensibilidade')
            axes[1,0].grid(True, alpha=0.3)
        else:
            axes[1,0].text(0.5,0.5,'Sem dados de sensibilidade', ha='center')

        # Especificidade
        if 'val_specificity' in self.densenet_history:
            axes[1,1].plot(epochs, self.densenet_history['val_specificity'], 'm-o', label='Especificidade', linewidth=2)
            axes[1,1].set_title('Especificidade (Validação)', fontsize=13, fontweight='bold')
            axes[1,1].set_xlabel('Época')
            axes[1,1].set_ylabel('Especificidade')
            axes[1,1].grid(True, alpha=0.3)
        else:
            axes[1,1].text(0.5,0.5,'Sem dados de especificidade', ha='center')

        plt.suptitle('DenseNet - Curvas de Aprendizado', fontsize=16, fontweight='bold')
        
        plt.tight_layout()
        plt.show()
    
    def analise_temporal_densenet(self):
        """Análise temporal para DenseNet - visitas posteriores têm idades maiores?"""
        if self.densenet_regress is None:
            messagebox.showwarning("Aviso", "Treine o DenseNet (regressão) primeiro!")
            return
        
        try:
            x_test = np.expand_dims(self.test_data[0], axis=-1)
            x_test = np.repeat(x_test, 3, axis=-1)
            
            y_pred = self.densenet_regress.predict(x_test, verbose=0).flatten()
            test_data_df = self.valid_data_df[self.valid_data_df['Subject ID'].isin(self.test_patients['Subject ID'])].copy()
            
            if len(y_pred) != len(test_data_df):
                messagebox.showwarning("Aviso", "Número de predições não corresponde aos dados de teste")
                return
            
            test_data_df['predicted_age'] = y_pred
            
            consistent_increase = 0
            total_patients = 0
            exemplos_bons = []
            exemplos_ruins = []
            diferencas_medias = []
            
            for subject_id, group in test_data_df.sort_values(['Subject ID', 'Visit']).groupby('Subject ID'):
                if len(group) > 1:
                    total_patients += 1
                    predicted_ages = group['predicted_age'].values
                    real_ages = group['Age'].values
                    
                    diffs_pred = np.diff(predicted_ages)
                    diffs_real = np.diff(real_ages)
                    
                    if np.all(diffs_pred >= 0):
                        consistent_increase += 1
                        if len(exemplos_bons) < 3:
                            exemplos_bons.append((subject_id, predicted_ages, real_ages))
                    else:
                        if len(exemplos_ruins) < 3:
                            exemplos_ruins.append((subject_id, predicted_ages, real_ages))
                    
                    diferencas_medias.append(np.mean(diffs_pred))
            
            ratio = (consistent_increase / total_patients * 100) if total_patients > 0 else 0
            
            result_text = (
                f"=== ANÁLISE TEMPORAL (DenseNet) ===\n\n"
                f"Pacientes com múltiplas visitas: {total_patients}\n"
                f"Com idades CRESCENTES: {consistent_increase}\n"
                f"Percentual de acerto: {ratio:.1f}%\n\n"
                f"Diferença média entre visitas: {np.mean(diferencas_medias):.2f} anos\n\n"
                f"RESPOSTA: {'SIM' if ratio >= 70 else 'NÃO'}, {ratio:.1f}% dos exames\n"
                f"em visitas posteriores resultaram em idades\n"
                f"{'maiores ou iguais' if ratio >= 70 else 'NEM SEMPRE maiores'}.\n\n"
                f"Comparação:\n"
                f"• DenseNet geralmente tem {'MELHOR' if ratio >= 70 else 'PIOR'} desempenho\n"
                f"  que XGBoost na progressão temporal.\n"
            )
            
            if exemplos_bons:
                result_text += "\nExemplos de SUCESSO:\n"
                for subj_id, pred_ages, real_ages in exemplos_bons[:2]:
                    result_text += f"  {subj_id}: {pred_ages[0]:.1f}→{pred_ages[-1]:.1f} anos (Real: {real_ages[0]:.1f}→{real_ages[-1]:.1f})\n"
            
            messagebox.showinfo("Análise Temporal - DenseNet", result_text)
            
            # Plotar gráfico comparativo
            if exemplos_bons:
                fig, axes = plt.subplots(1, min(3, len(exemplos_bons)), figsize=(15, 4))
                if len(exemplos_bons) == 1:
                    axes = [axes]
                
                for idx, (subj_id, pred_ages, real_ages) in enumerate(exemplos_bons[:3]):
                    ax = axes[idx] if len(exemplos_bons) > 1 else axes[0]
                    visitas = range(1, len(pred_ages) + 1)
                    
                    ax.plot(visitas, real_ages, 'g-o', label='Idade Real', linewidth=2, markersize=8)
                    ax.plot(visitas, pred_ages, 'r--s', label='Idade Predita (DenseNet)', linewidth=2, markersize=8)
                    ax.set_xlabel('Visita', fontsize=11)
                    ax.set_ylabel('Idade (anos)', fontsize=11)
                    ax.set_title(f'Paciente {subj_id}\n(DenseNet - Exemplo de Sucesso)', fontsize=12, fontweight='bold')
                    ax.legend()
                    ax.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.suptitle('Progressão Temporal - DenseNet (Exemplos Corretos)', 
                           fontsize=14, fontweight='bold', y=1.02)
                plt.show()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
            import traceback
            traceback.print_exc()
    
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
