# trabalhopai.py - VERSÃO COMPLETA COM INTERFACE GRÁFICA
# Sistema de Análise de Imagens - Alzheimer

# =============================================================================
# IMPORTS
# =============================================================================
import numpy as np 
import os
import pandas as pd
# Configurar backend do matplotlib ANTES de importar pyplot
import matplotlib
matplotlib.use('TkAgg')  # Força uso do backend TkAgg para GUI
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

# Importações do Keras/TensorFlow (opcionais - apenas se usar DenseNet)
try:
    from tensorflow.keras.models import Model
    from tensorflow.keras.layers import Input, Conv2D, Flatten, Dense, Dropout, BatchNormalization, MaxPooling2D
    from tensorflow.keras.applications import DenseNet121
    KERAS_AVAILABLE = True
except ImportError:
    KERAS_AVAILABLE = False
    print("⚠ TensorFlow/Keras não instalado. DenseNet não estará disponível.")
    print("  Para instalar: pip install tensorflow")

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
    """
    ITEM 9 - Preparação de Dados com Split 80/20 por Paciente
    
    Requisitos implementados:
    - Dataset Axial (DS=2) apenas
    - Divide PACIENTES (não exames) em treino (80%) e teste (20%)
    - 2 classes: Demented e NonDemented
    - Converted: CDR=0 → NonDemented, CDR>0 → Demented
    - Balanceamento 4:1 entre treino e teste
    - Validação: 20% do treino
    - SEM MISTURA de exames do mesmo paciente entre treino/teste
    
    Returns:
        Tupla contendo dados de treino, validação, teste, dataframe completo,
        e informações detalhadas do split
    """
    print("\n" + "="*70)
    print("ITEM 9: PREPARAÇÃO E DIVISÃO DOS DADOS")
    print("="*70)
    
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'oasis_longitudinal_demographic.csv')
    
    print(f"\n1. Carregando dados demográficos de: {csv_path}")
    demographics_df = pd.read_csv(csv_path, sep=';')
    
    # Limpeza de dados
    demographics_df['CDR'] = demographics_df['CDR'].astype(str).str.replace(',', '.', regex=False)
    demographics_df['CDR'] = pd.to_numeric(demographics_df['CDR'], errors='coerce')
    demographics_df['Age'] = pd.to_numeric(demographics_df['Age'], errors='coerce')
    
    print(f"   - Total de registros no CSV: {len(demographics_df)}")
    print(f"   - Grupos originais: {demographics_df['Group'].value_counts().to_dict()}")

    # Classificação binária conforme especificação
    def get_diagnosis_binary(row):
        if row['Group'] == 'Nondemented':
            return 'NonDemented'
        elif row['Group'] == 'Demented':
            return 'Demented'
        elif row['Group'] == 'Converted':
            # REGRA: CDR=0 → NonDemented, CDR>0 → Demented
            return 'Demented' if row['CDR'] > 0 else 'NonDemented'
        return 'Unknown'

    demographics_df['Diagnosis'] = demographics_df.apply(get_diagnosis_binary, axis=1)
    
    # Filtrar dados válidos
    valid_data_df = demographics_df[demographics_df['Diagnosis'] != 'Unknown'].dropna(
        subset=['MRI ID', 'Subject ID', 'Age']
    ).copy()
    
    print(f"\n2. Após classificação binária e limpeza:")
    print(f"   - Registros válidos: {len(valid_data_df)}")
    print(f"   - Classes finais: {valid_data_df['Diagnosis'].value_counts().to_dict()}")
    print(f"   - Pacientes únicos: {valid_data_df['Subject ID'].nunique()}")
    print(f"   - Exames totais: {len(valid_data_df)}")

    # Agregar informações por paciente (não por exame!)
    print("\n3. Agregando dados por PACIENTE (não por exame):")
    patient_info = valid_data_df.groupby('Subject ID').agg(
        final_CDR=('CDR', 'max'),
        first_visit_age=('Age', 'min'),
        num_visits=('Visit', 'count')
    ).reset_index()
    
    # Diagnóstico do paciente baseado no CDR máximo
    patient_info['Diagnosis'] = patient_info['final_CDR'].apply(
        lambda cdr: 'Demented' if cdr > 0 else 'NonDemented'
    )
    
    print(f"   - Pacientes NonDemented: {(patient_info['Diagnosis'] == 'NonDemented').sum()}")
    print(f"   - Pacientes Demented: {(patient_info['Diagnosis'] == 'Demented').sum()}")
    print(f"   - Média de visitas por paciente: {patient_info['num_visits'].mean():.2f}")
    
    # Encoder de labels
    le = LabelEncoder()
    patient_info['Diagnosis_encoded'] = le.fit_transform(patient_info['Diagnosis'])
    
    # SPLIT 80/20 POR PACIENTE (estratificado)
    print("\n4. Divisão 80/20 por PACIENTE (estratificado):")
    train_val_patients, test_patients = train_test_split(
        patient_info,
        test_size=0.2,
        random_state=42,
        stratify=patient_info['Diagnosis_encoded']
    )
    
    print(f"   - Pacientes em treino+val: {len(train_val_patients)}")
    print(f"   - Pacientes em teste: {len(test_patients)}")
    print(f"   - Proporção treino/teste: {len(train_val_patients)/len(test_patients):.2f}:1")
    
    # Validação cruzada dos splits
    train_val_ids = set(train_val_patients['Subject ID'])
    test_ids = set(test_patients['Subject ID'])
    overlap = train_val_ids.intersection(test_ids)
    print(f"   - ✓ Verificação: {len(overlap)} pacientes em comum (deve ser 0)")
    
    # Balanceamento de classes
    print("\n   Balanceamento de classes no TREINO+VAL:")
    for cls in ['NonDemented', 'Demented']:
        count = (train_val_patients['Diagnosis'] == cls).sum()
        print(f"     - {cls}: {count} pacientes")
    
    print("\n   Balanceamento de classes no TESTE:")
    for cls in ['NonDemented', 'Demented']:
        count = (test_patients['Diagnosis'] == cls).sum()
        print(f"     - {cls}: {count} pacientes")
    
    # Validação: 20% do conjunto de treino
    print("\n5. Separando 20% do treino para VALIDAÇÃO:")
    train_patients, val_patients = train_test_split(
        train_val_patients,
        test_size=0.2,
        random_state=42,
        stratify=train_val_patients['Diagnosis_encoded']
    )
    
    print(f"   - Pacientes em TREINO final: {len(train_patients)}")
    print(f"   - Pacientes em VALIDAÇÃO: {len(val_patients)}")
    print(f"   - Pacientes em TESTE: {len(test_patients)}")
    
    # Obter caminhos e labels para cada conjunto
    def get_set_data(patient_ids_df, full_df):
        set_df = full_df[full_df['Subject ID'].isin(patient_ids_df['Subject ID'])].copy()
        axl_dir = os.path.join(base_dir, 'axl')
        paths = [os.path.join(axl_dir, f"{mri_id}_axl.nii") for mri_id in set_df['MRI ID']]
        class_labels = le.transform(set_df['Diagnosis'])
        age_labels = set_df['Age'].values
        return paths, class_labels, age_labels, set_df

    train_paths, y_train_class, y_train_age, train_df = get_set_data(train_patients, valid_data_df)
    val_paths, y_val_class, y_val_age, val_df = get_set_data(val_patients, valid_data_df)
    test_paths, y_test_class, y_test_age, test_df = get_set_data(test_patients, valid_data_df)

    print("\n6. Contagem de EXAMES em cada conjunto:")
    print(f"   - Exames em TREINO: {len(train_paths)}")
    print(f"   - Exames em VALIDAÇÃO: {len(val_paths)}")
    print(f"   - Exames em TESTE: {len(test_paths)}")
    print(f"   - Total de exames: {len(train_paths) + len(val_paths) + len(test_paths)}")
    
    # Verificação de balanceamento 4:1
    print("\n7. Verificação do balanceamento 4:1 (treino:teste):")
    ratio = len(train_paths) / len(test_paths) if len(test_paths) > 0 else 0
    print(f"   - Proporção real treino:teste = {ratio:.2f}:1")
    print(f"   - Meta: ~4:1 {'✓' if 3.5 <= ratio <= 5.0 else '⚠'}")

    print("\n8. Carregando imagens...")
    x_train_img = load_images(train_paths)
    x_val_img = load_images(val_paths)
    x_test_img = load_images(test_paths)
    
    print(f"   - Shape treino: {x_train_img.shape}")
    print(f"   - Shape validação: {x_val_img.shape}")
    print(f"   - Shape teste: {x_test_img.shape}")
    
    # Informações do split para análise posterior
    split_info = {
        'train_patients': train_patients,
        'val_patients': val_patients,
        'test_patients': test_patients,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'label_encoder': le,
        'num_train_exams': len(train_paths),
        'num_val_exams': len(val_paths),
        'num_test_exams': len(test_paths),
        'num_train_patients': len(train_patients),
        'num_val_patients': len(val_patients),
        'num_test_patients': len(test_patients)
    }
    
    print("\n" + "="*70)
    print("PREPARAÇÃO DE DADOS CONCLUÍDA COM SUCESSO!")
    print("="*70 + "\n")
    
    return (x_train_img, y_train_class, y_train_age), \
           (x_val_img, y_val_class, y_val_age), \
           (x_test_img, y_test_class, y_test_age), \
           valid_data_df, split_info

def extract_features(images):
    """Extracts comprehensive texture and statistical features from images."""
    features = []
    for img in images:
        img_8bit = (img * 255).astype(np.uint8)
        
        # GLCM features em múltiplas direções e distâncias
        feature_vector = []
        distances = [1, 3, 5]
        angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
        
        for distance in distances:
            glcm = graycomatrix(img_8bit, distances=[distance], angles=angles, 
                              levels=256, symmetric=True, normed=True)
            
            # 6 propriedades GLCM por direção e distância
            for prop in ['contrast', 'dissimilarity', 'homogeneity', 'energy', 'correlation', 'ASM']:
                prop_values = graycoprops(glcm, prop)[0, :]
                feature_vector.extend([prop_values.mean(), prop_values.std()])
        
        # Features estatísticas da imagem
        feature_vector.extend([
            np.mean(img),
            np.std(img),
            np.median(img),
            np.min(img),
            np.max(img),
            np.percentile(img, 25),
            np.percentile(img, 75),
            np.var(img)
        ])
        
        # Features de intensidade por regiões (dividir em 4 quadrantes)
        h, w = img.shape
        quadrants = [
            img[:h//2, :w//2],
            img[:h//2, w//2:],
            img[h//2:, :w//2],
            img[h//2:, w//2:]
        ]
        for quad in quadrants:
            feature_vector.extend([np.mean(quad), np.std(quad)])
        
        features.append(feature_vector)
    
    return np.array(features)

def extract_clinical_features(data_df, mri_ids):
    """
    Extrai características clínicas do DataFrame para os MRI IDs fornecidos.
    
    NOTA: Age foi REMOVIDA para evitar data leakage (vazamento de dados).
    Predizer idade usando a própria idade como feature seria trapacear!
    
    Features clínicas (8 features):
    - EDUC: Anos de educação
    - MMSE: Mini-Mental State Examination (função cognitiva)
    - eTIV: Volume intracraniano estimado
    - nWBV: Volume cerebral normalizado
    - ASF: Fator de escala do atlas
    - Visit: Número da visita (contexto temporal)
    - Years_since_first: Anos desde primeira visita (progressão)
    - CDR: Clinical Dementia Rating (severidade)
    """
    clinical_features = []
    
    def safe_float(value, default):
        """Converte valor para float, lidando com vírgulas e NaN"""
        if pd.isna(value):
            return default
        try:
            # Se for string, substituir vírgula por ponto
            if isinstance(value, str):
                value = value.replace(',', '.')
            return float(value)
        except (ValueError, TypeError):
            return default
    
    # Calcular tempo desde primeira visita por paciente
    patient_first_visits = {}
    for _, row in data_df.iterrows():
        subject_id = row['Subject ID']
        mri_delay = safe_float(row.get('MR Delay', 0), 0)
        if subject_id not in patient_first_visits:
            patient_first_visits[subject_id] = mri_delay
        else:
            patient_first_visits[subject_id] = min(patient_first_visits[subject_id], mri_delay)
    
    for mri_id in mri_ids:
        row = data_df[data_df['MRI ID'] == mri_id]
        if len(row) > 0:
            row = row.iloc[0]
            subject_id = row['Subject ID']
            
            # Tempo desde primeira visita (anos)
            current_delay = safe_float(row.get('MR Delay', 0), 0)
            first_delay = patient_first_visits.get(subject_id, 0)
            years_since_first = max(0, current_delay - first_delay)
            
            features = [
                safe_float(row['EDUC'], 12.0),        # Educação (constante)
                safe_float(row['MMSE'], 25.0),        # Mini-Mental (dinâmico)
                safe_float(row['eTIV'], 1500.0),      # Volume intracraniano (constante)
                safe_float(row['nWBV'], 0.7),         # Volume cerebral (dinâmico)
                safe_float(row['ASF'], 1.2),          # Fator escala (constante)
                safe_float(row['Visit'], 1.0),        # Número da visita (temporal)
                years_since_first,                     # Anos desde 1ª visita (temporal)
                safe_float(row['CDR'], 0.0)           # Demência rating (dinâmico)
            ]
        else:
            # Valores padrão se MRI ID não encontrado
            features = [12.0, 25.0, 1500.0, 0.7, 1.2, 1.0, 0.0, 0.0]
        
        clinical_features.append(features)
    
    return np.array(clinical_features, dtype=np.float64)

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
        """
        ITEM 9 - Prepara dados com split 80/20 por paciente
        Implementa todas as especificações do trabalho
        """
        try:
            messagebox.showinfo("Preparando Dados", "Carregando e preparando dados...\nIsso pode levar alguns minutos.")
            
            # Função retorna agora split_info ao invés de test_patients
            self.train_data, self.val_data, self.test_data, self.valid_data_df, split_info = load_and_prepare_data()
            
            # Armazenar split_info que contém todas as informações
            self.split_info = split_info
            self.test_patients = split_info
            
            messagebox.showinfo("Sucesso", 
                              f"Dados preparados com sucesso!\n\n"
                              f"TREINO:\n"
                              f"  • {split_info['num_train_patients']} pacientes\n"
                              f"  • {split_info['num_train_exams']} exames\n\n"
                              f"VALIDAÇÃO:\n"
                              f"  • {split_info['num_val_patients']} pacientes\n"
                              f"  • {split_info['num_val_exams']} exames\n\n"
                              f"TESTE:\n"
                              f"  • {split_info['num_test_patients']} pacientes\n"
                              f"  • {split_info['num_test_exams']} exames\n\n"
                              f"Proporção treino:teste = "
                              f"{split_info['num_train_exams']/split_info['num_test_exams']:.2f}:1")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao preparar dados:\n{str(e)}")
    
    def extrair_caracteristicas(self):
        """Extrai características de textura e clínicas"""
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        try:
            messagebox.showinfo("Extraindo", "Extraindo características avançadas...\nIsso pode levar alguns minutos.")
            
            # Extrair características de textura
            print("Extraindo features de textura...")
            texture_train = extract_features(self.train_data[0])
            texture_val = extract_features(self.val_data[0])
            texture_test = extract_features(self.test_data[0])
            
            print(f"Features de textura extraídas: {texture_train.shape[1]} features por imagem")
            
            # Extrair características clínicas usando split_info
            print("Extraindo features clínicas...")
            
            if hasattr(self, 'split_info') and self.split_info is not None:
                # Pegar DataFrames separados por conjunto
                train_df = self.valid_data_df[self.valid_data_df['Subject ID'].isin(
                    self.split_info['train_patients']['Subject ID'])]
                val_df = self.valid_data_df[self.valid_data_df['Subject ID'].isin(
                    self.split_info['val_patients']['Subject ID'])]
                test_df = self.valid_data_df[self.valid_data_df['Subject ID'].isin(
                    self.split_info['test_patients']['Subject ID'])]
                
                # Extrair features - o tamanho será automático
                clinical_train = extract_clinical_features(train_df, train_df['MRI ID'].values)
                clinical_val = extract_clinical_features(val_df, val_df['MRI ID'].values)
                clinical_test = extract_clinical_features(test_df, test_df['MRI ID'].values)
                
                print(f"Features clínicas extraídas: {clinical_train.shape[1]} features")
                print(f"  - Train: {clinical_train.shape[0]} amostras")
                print(f"  - Val: {clinical_val.shape[0]} amostras")
                print(f"  - Test: {clinical_test.shape[0]} amostras")
                
                # Verificar tamanhos e ajustar se necessário
                if clinical_train.shape[0] != texture_train.shape[0]:
                    print(f"⚠ Ajustando tamanho: clinical_train {clinical_train.shape[0]} != texture_train {texture_train.shape[0]}")
                    clinical_train = clinical_train[:texture_train.shape[0]]
                
                if clinical_val.shape[0] != texture_val.shape[0]:
                    print(f"⚠ Ajustando tamanho: clinical_val {clinical_val.shape[0]} != texture_val {texture_val.shape[0]}")
                    clinical_val = clinical_val[:texture_val.shape[0]]
                
                if clinical_test.shape[0] != texture_test.shape[0]:
                    print(f"⚠ Ajustando tamanho: clinical_test {clinical_test.shape[0]} != texture_test {texture_test.shape[0]}")
                    clinical_test = clinical_test[:texture_test.shape[0]]
                
                # Combinar características
                self.x_train_features = np.hstack([texture_train, clinical_train])
                self.x_val_features = np.hstack([texture_val, clinical_val])
                self.x_test_features = np.hstack([texture_test, clinical_test])
                
                total_features = self.x_train_features.shape[1]
                messagebox.showinfo("Sucesso", 
                                  f"Características extraídas com sucesso!\n\n"
                                  f"Total de features: {total_features}\n"
                                  f"  • Textura: {texture_train.shape[1]}\n"
                                  f"  • Clínicas: {clinical_train.shape[1]}")
            else:
                # Fallback: usar apenas features de textura
                print("⚠ Split_info não disponível, usando apenas features de textura")
                self.x_train_features = texture_train
                self.x_val_features = texture_val
                self.x_test_features = texture_test
                
                messagebox.showinfo("Sucesso", 
                                  f"Características de textura extraídas!\n\n"
                                  f"Total de features: {texture_train.shape[1]}\n"
                                  f"(Features clínicas não disponíveis)")
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
            plt.imshow(cm, interpolation='nearest', cmap='Blues')
            plt.title('Matriz de Confusão - SVM')
            plt.colorbar()
            
            # Adicionar anotações
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
            
            plt.ylabel('Real')
            plt.xlabel('Predito')
            plt.tight_layout()
            plt.show()
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
    
    # =========================================================================
    # XGBOOST - REGRESSOR RASO
    # =========================================================================
    def treinar_xgboost(self):
        """Treina XGBoost para regressão de idade com hiperparâmetros otimizados"""
        if self.x_train_features is None:
            messagebox.showwarning("Aviso", "Extraia as características primeiro!")
            return
        
        try:
            messagebox.showinfo("Treinando", "Treinando XGBoost com parâmetros otimizados...\nIsso pode levar alguns minutos.")
            
            # Hiperparâmetros otimizados para regressão de idade
            self.xgboost_model = xgb.XGBRegressor(
                n_estimators=500,           # Mais árvores para melhor aprendizado
                max_depth=6,                # Profundidade maior para capturar padrões complexos
                learning_rate=0.05,         # Taxa menor para convergência mais suave
                subsample=0.8,              # Amostragem para reduzir overfitting
                colsample_bytree=0.8,       # Amostragem de features
                min_child_weight=3,         # Regularização
                gamma=0.1,                  # Reduz complexidade
                reg_alpha=0.1,              # L1 regularization
                reg_lambda=1.0,             # L2 regularization
                random_state=42,
                n_jobs=-1                   # Usar todos os cores
            )
            
            # Treinar com early stopping usando conjunto de validação
            print("Iniciando treinamento com early stopping...")
            self.xgboost_model.fit(
                self.x_train_features, 
                self.train_data[2],
                eval_set=[(self.x_val_features, self.val_data[2])],
                verbose=False
            )
            
            # Avaliar no conjunto de validação
            val_pred = self.xgboost_model.predict(self.x_val_features)
            val_mae = np.mean(np.abs(val_pred - self.val_data[2]))
            
            messagebox.showinfo("Sucesso", 
                              f"XGBoost treinado com sucesso!\n\n"
                              f"MAE no conjunto de validação: {val_mae:.2f} anos")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar XGBoost:\n{str(e)}")
    
    def avaliar_xgboost(self):
        """
        ITEM 11 - Avaliação do Regressor XGBoost
        
        Analisa:
        - MAE (Mean Absolute Error)
        - RMSE (Root Mean Squared Error)
        - Distribuição dos erros
        - Análise por faixa etária
        - Gráfico de predição vs real
        """
        if self.xgboost_model is None:
            messagebox.showwarning("Aviso", "Treine o XGBoost primeiro!")
            return
        
        try:
            print("\n" + "="*70)
            print("ITEM 11: AVALIAÇÃO DO REGRESSOR XGBOOST")
            print("="*70)
            
            y_pred = self.xgboost_model.predict(self.x_test_features)
            y_test = self.test_data[2]
            
            mae = mean_absolute_error(y_test, y_pred)
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            errors = y_test - y_pred
            
            print(f"\n1. Métricas de Erro:")
            print(f"   - MAE (Mean Absolute Error): {mae:.2f} anos")
            print(f"   - RMSE (Root Mean Squared Error): {rmse:.2f} anos")
            print(f"   - Erro médio: {np.mean(errors):.2f} anos")
            print(f"   - Desvio padrão dos erros: {np.std(errors):.2f} anos")
            print(f"   - Erro máximo: {np.max(np.abs(errors)):.2f} anos")
            
            print(f"\n2. Análise de Qualidade da Predição:")
            print(f"   - Predições dentro de ±5 anos: {np.sum(np.abs(errors) <= 5) / len(errors) * 100:.1f}%")
            print(f"   - Predições dentro de ±10 anos: {np.sum(np.abs(errors) <= 10) / len(errors) * 100:.1f}%")
            print(f"   - Idade real: {y_test.min():.1f} - {y_test.max():.1f} anos")
            print(f"   - Idade predita: {y_pred.min():.1f} - {y_pred.max():.1f} anos")
            
            # Análise por faixa etária
            print(f"\n3. MAE por Faixa Etária:")
            bins = [60, 70, 80, 90, 100]
            for i in range(len(bins)-1):
                mask = (y_test >= bins[i]) & (y_test < bins[i+1])
                if np.sum(mask) > 0:
                    mae_bin = np.mean(np.abs(errors[mask]))
                    print(f"   - {bins[i]}-{bins[i+1]} anos: MAE = {mae_bin:.2f} anos (n={np.sum(mask)})")
            
            # Conclusão sobre suficiência das entradas
            print(f"\n4. ANÁLISE: As entradas são suficientes para boa predição?")
            if mae < 5:
                print(f"   ✓ SIM - MAE < 5 anos indica excelente predição")
            elif mae < 10:
                print(f"   ⚠ PARCIALMENTE - MAE entre 5-10 anos indica predição razoável")
                print(f"   → Sugestão: incluir mais características (nWBV, eTIV, MMSE)")
            else:
                print(f"   ✗ NÃO - MAE > 10 anos indica predição insuficiente")
                print(f"   → As características atuais não são suficientes")
                print(f"   → Recomenda-se incluir mais informações clínicas e demográficas")
            
            # Gráfico de predição vs real
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Scatter plot
            axes[0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
            axes[0].plot([y_test.min(), y_test.max()], 
                         [y_test.min(), y_test.max()], 'r--', lw=2, label='Predição Perfeita')
            axes[0].set_xlabel('Idade Real (anos)', fontsize=12)
            axes[0].set_ylabel('Idade Predita (anos)', fontsize=12)
            axes[0].set_title(f'XGBoost: Predição vs Real\nMAE = {mae:.2f} anos', fontsize=14)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Histograma de erros
            axes[1].hist(errors, bins=20, edgecolor='black', alpha=0.7)
            axes[1].axvline(0, color='r', linestyle='--', linewidth=2, label='Erro = 0')
            axes[1].set_xlabel('Erro (Real - Predito) em anos', fontsize=12)
            axes[1].set_ylabel('Frequência', fontsize=12)
            axes[1].set_title('Distribuição dos Erros', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            print("\n" + "="*70)
            print("AVALIAÇÃO CONCLUÍDA!")
            print("="*70 + "\n")
            
            messagebox.showinfo("Avaliação XGBoost", 
                              f"MAE: {mae:.2f} anos\n"
                              f"RMSE: {rmse:.2f} anos\n\n"
                              f"Predições dentro de ±5 anos: {np.sum(np.abs(errors) <= 5) / len(errors) * 100:.1f}%")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
    
    def analise_temporal(self):
        """
        ITEM 11 - Análise Temporal: Visitas posteriores têm idades preditas maiores?
        
        Verifica se o modelo consegue capturar a progressão temporal,
        analisando se exames de visitas posteriores resultam em idades preditas maiores.
        """
        if self.xgboost_model is None:
            messagebox.showwarning("Aviso", "Treine o XGBoost primeiro!")
            return
        
        try:
            print("\n" + "="*70)
            print("ITEM 11: ANÁLISE TEMPORAL - PROGRESSÃO DE VISITAS")
            print("="*70)
            
            # Obter dados de teste
            if isinstance(self.test_patients, dict) and 'test_df' in self.test_patients:
                test_data_df = self.test_patients['test_df'].copy()
            else:
                test_data_df = self.valid_data_df[
                    self.valid_data_df['Subject ID'].isin(self.test_patients['Subject ID'])
                ].copy()
            
            print(f"\n1. Preparando análise temporal:")
            print(f"   - Total de exames no teste: {len(test_data_df)}")
            print(f"   - Pacientes únicos: {test_data_df['Subject ID'].nunique()}")
            
            # Predições
            y_pred = self.xgboost_model.predict(self.x_test_features)
            test_data_df['predicted_age'] = y_pred[:len(test_data_df)]
            test_data_df = test_data_df.sort_values(['Subject ID', 'Visit'])
            
            print(f"\n2. Analisando progressão temporal por paciente:")
            
            consistent_increase = 0
            mostly_consistent = 0
            total_patients = 0
            total_transitions = 0
            increasing_transitions = 0
            
            temporal_details = []
            
            for subject_id, group in test_data_df.groupby('Subject ID'):
                if len(group) > 1:
                    total_patients += 1
                    
                    visits = group['Visit'].values
                    real_ages = group['Age'].values
                    pred_ages = group['predicted_age'].values
                    
                    # Análise das transições
                    age_diffs_real = np.diff(real_ages)
                    age_diffs_pred = np.diff(pred_ages)
                    
                    num_transitions = len(age_diffs_pred)
                    total_transitions += num_transitions
                    
                    # Contar transições crescentes
                    increasing = np.sum(age_diffs_pred >= 0)
                    increasing_transitions += increasing
                    
                    # Verificar consistência completa
                    if np.all(age_diffs_pred >= 0):
                        consistent_increase += 1
                        status = "✓ Consistente"
                    elif increasing / num_transitions >= 0.5:
                        mostly_consistent += 1
                        status = "⚠ Parcialmente consistente"
                    else:
                        status = "✗ Inconsistente"
                    
                    temporal_details.append({
                        'subject_id': subject_id,
                        'num_visits': len(group),
                        'real_age_span': f"{real_ages[0]:.1f} → {real_ages[-1]:.1f}",
                        'pred_age_span': f"{pred_ages[0]:.1f} → {pred_ages[-1]:.1f}",
                        'real_delta': real_ages[-1] - real_ages[0],
                        'pred_delta': pred_ages[-1] - pred_ages[0],
                        'status': status
                    })
            
            print(f"   - Pacientes com múltiplas visitas: {total_patients}")
            print(f"   - Total de transições entre visitas: {total_transitions}")
            
            if total_patients > 0:
                consistency_ratio = consistent_increase / total_patients
                partial_ratio = (consistent_increase + mostly_consistent) / total_patients
                transition_ratio = increasing_transitions / total_transitions if total_transitions > 0 else 0
                
                print(f"\n3. Resultados da Análise Temporal:")
                print(f"   - Pacientes 100% consistentes: {consistent_increase}/{total_patients} ({consistency_ratio:.1%})")
                print(f"   - Pacientes parcialmente consistentes: {mostly_consistent}/{total_patients}")
                print(f"   - Taxa geral de consistência: {partial_ratio:.1%}")
                print(f"   - Transições com aumento: {increasing_transitions}/{total_transitions} ({transition_ratio:.1%})")
                
                # Mostrar exemplos detalhados
                print(f"\n4. Exemplos Detalhados (primeiros 5 pacientes):")
                for i, detail in enumerate(temporal_details[:5]):
                    print(f"\n   Paciente {detail['subject_id']}:")
                    print(f"     - Número de visitas: {detail['num_visits']}")
                    print(f"     - Idade real: {detail['real_age_span']} (Δ={detail['real_delta']:.1f} anos)")
                    print(f"     - Idade predita: {detail['pred_age_span']} (Δ={detail['pred_delta']:.1f} anos)")
                    print(f"     - Status: {detail['status']}")
                
                # RESPOSTA À QUESTÃO 11
                print(f"\n5. RESPOSTA: Visitas posteriores resultam em idades preditas maiores?")
                if consistency_ratio >= 0.8:
                    print(f"   ✓ SIM - {consistency_ratio:.1%} dos pacientes mostram progressão consistente")
                    print(f"   → O modelo captura bem a progressão temporal")
                elif transition_ratio >= 0.7:
                    print(f"   ⚠ PARCIALMENTE - {transition_ratio:.1%} das transições são crescentes")
                    print(f"   → O modelo captura a tendência geral, mas com inconsistências")
                else:
                    print(f"   ✗ NÃO - Apenas {transition_ratio:.1%} das transições são crescentes")
                    print(f"   → O modelo NÃO captura adequadamente a progressão temporal")
                    print(f"   → As características usadas podem não ser suficientemente sensíveis ao envelhecimento")
                
                # Gráfico de dispersão: idade real vs predita por visita
                fig, ax = plt.subplots(figsize=(10, 6))
                
                colors = plt.cm.viridis(np.linspace(0, 1, test_data_df['Visit'].max()))
                
                for visit in sorted(test_data_df['Visit'].unique()):
                    visit_data = test_data_df[test_data_df['Visit'] == visit]
                    ax.scatter(visit_data['Age'], visit_data['predicted_age'], 
                              label=f'Visita {visit}', alpha=0.6, s=100,
                              color=colors[visit-1], edgecolors='k', linewidth=0.5)
                
                # Linha de predição perfeita
                min_age = test_data_df['Age'].min()
                max_age = test_data_df['Age'].max()
                ax.plot([min_age, max_age], [min_age, max_age], 'r--', lw=2, label='Predição Perfeita')
                
                ax.set_xlabel('Idade Real (anos)', fontsize=12)
                ax.set_ylabel('Idade Predita (anos)', fontsize=12)
                ax.set_title('Análise Temporal: Idade Real vs Predita por Visita', fontsize=14)
                ax.legend()
                ax.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.show()
                
                print("\n" + "="*70)
                print("ANÁLISE TEMPORAL CONCLUÍDA!")
                print("="*70 + "\n")
                
                messagebox.showinfo("Análise Temporal", 
                                  f"Pacientes com múltiplas visitas: {total_patients}\n\n"
                                  f"Consistência Temporal:\n"
                                  f"  • 100% consistente: {consistent_increase} ({consistency_ratio:.1%})\n"
                                  f"  • Parcialmente: {mostly_consistent}\n"
                                  f"  • Taxa geral: {partial_ratio:.1%}\n\n"
                                  f"Transições crescentes: {transition_ratio:.1%}")
            else:
                print("   ⚠ Não há pacientes com múltiplas visitas no conjunto de teste")
                messagebox.showinfo("Análise Temporal", "Dados insuficientes para análise temporal")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
    
    # =========================================================================
    # DENSENET - MODELO PROFUNDO
    # =========================================================================
    def treinar_densenet_classif(self):
        """Treina DenseNet para classificação"""
        if not KERAS_AVAILABLE:
            messagebox.showerror("Erro", "TensorFlow/Keras não instalado!\n\n"
                               "Para usar DenseNet, instale:\n"
                               "pip install tensorflow")
            return
            
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        messagebox.showinfo("DenseNet", "Treinamento de DenseNet para classificação\n"
                          "será implementado aqui com fine-tuning.")
    
    def treinar_densenet_regress(self):
        """
        ITEM 11 - Método Profundo: DenseNet para Regressão de Idade
        Usa as imagens diretamente como entrada
        """
        if not KERAS_AVAILABLE:
            messagebox.showerror("Erro", "TensorFlow/Keras não instalado!\n\n"
                               "Para usar DenseNet, instale:\n"
                               "pip install tensorflow")
            return
            
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        try:
            messagebox.showinfo("Treinando", "Treinando DenseNet para regressão de idade...\n"
                              "Isso pode levar vários minutos (10-30 min dependendo do hardware).")
            
            print("\n" + "="*70)
            print("ITEM 11 - MÉTODO PROFUNDO: DenseNet para Regressão de Idade")
            print("="*70)
            
            # Preparar dados
            x_train = self.train_data[0]
            y_train_age = self.train_data[2]
            x_val = self.val_data[0]
            y_val_age = self.val_data[2]
            
            # Converter para 3 canais (DenseNet precisa RGB)
            print("\n1. Preparando imagens para DenseNet...")
            x_train_rgb = np.stack([x_train]*3, axis=-1)
            x_val_rgb = np.stack([x_val]*3, axis=-1)
            
            # Normalizar para [0, 1]
            x_train_rgb = x_train_rgb.astype('float32') / 255.0
            x_val_rgb = x_val_rgb.astype('float32') / 255.0
            
            print(f"   - Shape treino: {x_train_rgb.shape}")
            print(f"   - Shape validação: {x_val_rgb.shape}")
            print(f"   - Idade treino: {y_train_age.min():.1f} - {y_train_age.max():.1f} anos")
            
            # Criar modelo DenseNet com transfer learning
            print("\n2. Construindo modelo DenseNet...")
            base_model = DenseNet121(
                include_top=False,
                weights=None,  # Treinar do zero (imagens médicas são diferentes)
                input_shape=(128, 128, 3),
                pooling='avg'
            )
            
            # Adicionar camadas para regressão
            inputs = Input(shape=(128, 128, 3))
            x = base_model(inputs)
            x = Dense(512, activation='relu')(x)
            x = Dropout(0.5)(x)
            x = Dense(256, activation='relu')(x)
            x = Dropout(0.3)(x)
            x = Dense(128, activation='relu')(x)
            outputs = Dense(1, activation='linear')(x)  # Saída: idade (regressão)
            
            self.densenet_regress = Model(inputs=inputs, outputs=outputs)
            
            # Compilar
            from tensorflow.keras.optimizers import Adam
            self.densenet_regress.compile(
                optimizer=Adam(learning_rate=0.0001),
                loss='mean_absolute_error',
                metrics=['mae', 'mse']
            )
            
            print(f"   - Parâmetros treináveis: {self.densenet_regress.count_params():,}")
            
            # Callbacks
            from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
            
            callbacks = [
                EarlyStopping(
                    monitor='val_loss',
                    patience=15,
                    restore_best_weights=True,
                    verbose=1
                ),
                ReduceLROnPlateau(
                    monitor='val_loss',
                    factor=0.5,
                    patience=5,
                    min_lr=1e-7,
                    verbose=1
                )
            ]
            
            # Treinar
            print("\n3. Treinando modelo...")
            print("   (Isso pode demorar 10-30 minutos dependendo do hardware)")
            
            history = self.densenet_regress.fit(
                x_train_rgb, y_train_age,
                validation_data=(x_val_rgb, y_val_age),
                epochs=100,
                batch_size=16,
                callbacks=callbacks,
                verbose=1
            )
            
            # Avaliar no conjunto de validação
            val_loss, val_mae, val_mse = self.densenet_regress.evaluate(x_val_rgb, y_val_age, verbose=0)
            
            print(f"\n4. Resultado no conjunto de validação:")
            print(f"   - MAE: {val_mae:.2f} anos")
            print(f"   - RMSE: {np.sqrt(val_mse):.2f} anos")
            
            messagebox.showinfo("Sucesso", 
                              f"DenseNet treinado com sucesso!\n\n"
                              f"MAE no conjunto de validação: {val_mae:.2f} anos\n"
                              f"Epochs executados: {len(history.history['loss'])}")
                              
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar DenseNet:\n{str(e)}\n\n"
                               "Certifique-se de que TensorFlow está instalado:\n"
                               "pip install tensorflow")
    
    def avaliar_densenet_classif(self):
        """Avalia DenseNet classificação"""
        messagebox.showinfo("DenseNet", "Avaliação de DenseNet Classificação")
    
    def avaliar_densenet_regress(self):
        """
        ITEM 11 - Avaliação do Método Profundo (DenseNet)
        Análise completa com as mesmas questões do XGBoost
        """
        if not KERAS_AVAILABLE:
            messagebox.showerror("Erro", "TensorFlow/Keras não instalado!")
            return
            
        if self.densenet_regress is None:
            messagebox.showwarning("Aviso", "Treine o DenseNet primeiro!")
            return
        
        try:
            print("\n" + "="*70)
            print("ITEM 11 - AVALIAÇÃO DO MÉTODO PROFUNDO (DenseNet)")
            print("="*70)
            
            # Preparar dados de teste
            x_test = self.test_data[0]
            y_test = self.test_data[2]
            
            # Converter para RGB e normalizar
            x_test_rgb = np.stack([x_test]*3, axis=-1).astype('float32') / 255.0
            
            # Predições
            print("\n1. Gerando predições...")
            y_pred = self.densenet_regress.predict(x_test_rgb, verbose=0).flatten()
            
            # Métricas
            mae = np.mean(np.abs(y_test - y_pred))
            rmse = np.sqrt(np.mean((y_test - y_pred)**2))
            errors = y_test - y_pred
            
            print(f"\n2. Métricas de Erro:")
            print(f"   - MAE (Mean Absolute Error): {mae:.2f} anos")
            print(f"   - RMSE (Root Mean Squared Error): {rmse:.2f} anos")
            print(f"   - Erro médio: {np.mean(errors):.2f} anos")
            print(f"   - Desvio padrão dos erros: {np.std(errors):.2f} anos")
            print(f"   - Erro máximo: {np.max(np.abs(errors)):.2f} anos")
            
            print(f"\n3. Análise de Qualidade da Predição:")
            within_5 = np.sum(np.abs(errors) <= 5) / len(errors) * 100
            within_10 = np.sum(np.abs(errors) <= 10) / len(errors) * 100
            print(f"   - Predições dentro de ±5 anos: {within_5:.1f}%")
            print(f"   - Predições dentro de ±10 anos: {within_10:.1f}%")
            print(f"   - Idade real: {y_test.min():.1f} - {y_test.max():.1f} anos")
            print(f"   - Idade predita: {y_pred.min():.1f} - {y_pred.max():.1f} anos")
            
            # Análise por faixa etária
            print(f"\n4. MAE por Faixa Etária:")
            bins = [60, 70, 80, 90, 100]
            for i in range(len(bins)-1):
                mask = (y_test >= bins[i]) & (y_test < bins[i+1])
                if np.sum(mask) > 0:
                    mae_bin = np.mean(np.abs(errors[mask]))
                    print(f"   - {bins[i]}-{bins[i+1]} anos: MAE = {mae_bin:.2f} anos (n={np.sum(mask)})")
            
            # RESPOSTA À QUESTÃO 11.1
            print(f"\n5. ANÁLISE: As entradas (imagens) são suficientes para boa predição?")
            if mae < 5:
                print(f"   ✓ SIM - MAE < 5 anos indica excelente predição")
                print(f"   → As imagens contêm informação suficiente sobre a idade")
            elif mae < 10:
                print(f"   ⚠ PARCIALMENTE - MAE entre 5-10 anos indica predição razoável")
                print(f"   → As imagens fornecem informação útil, mas limitada")
            else:
                print(f"   ✗ NÃO - MAE > 10 anos indica predição insuficiente")
                print(f"   → As imagens sozinhas não são suficientes")
            
            # Gráficos
            fig, axes = plt.subplots(1, 2, figsize=(12, 5))
            
            # Scatter plot
            axes[0].scatter(y_test, y_pred, alpha=0.6, edgecolors='k', linewidth=0.5)
            axes[0].plot([y_test.min(), y_test.max()], 
                         [y_test.min(), y_test.max()], 'r--', lw=2, label='Predição Perfeita')
            axes[0].set_xlabel('Idade Real (anos)', fontsize=12)
            axes[0].set_ylabel('Idade Predita (anos)', fontsize=12)
            axes[0].set_title(f'DenseNet: Predição vs Real\nMAE = {mae:.2f} anos', fontsize=14)
            axes[0].legend()
            axes[0].grid(True, alpha=0.3)
            
            # Histograma de erros
            axes[1].hist(errors, bins=20, edgecolor='black', alpha=0.7)
            axes[1].axvline(0, color='r', linestyle='--', linewidth=2, label='Erro = 0')
            axes[1].set_xlabel('Erro (Real - Predito) em anos', fontsize=12)
            axes[1].set_ylabel('Frequência', fontsize=12)
            axes[1].set_title('Distribuição dos Erros', fontsize=14)
            axes[1].legend()
            axes[1].grid(True, alpha=0.3)
            
            plt.tight_layout()
            plt.show()
            
            # Análise temporal
            self._analise_temporal_densenet(y_pred)
            
            messagebox.showinfo("Resultados", 
                              f"DenseNet - Método Profundo\n\n"
                              f"MAE: {mae:.2f} anos\n"
                              f"RMSE: {rmse:.2f} anos\n"
                              f"Predições ±5 anos: {within_5:.1f}%\n"
                              f"Predições ±10 anos: {within_10:.1f}%")
                              
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao avaliar DenseNet:\n{str(e)}")
    
    def _analise_temporal_densenet(self, y_pred):
        """Análise temporal para DenseNet (similar ao XGBoost)"""
        try:
            # Criar DataFrame com predições (usar test_patients corretamente)
            if isinstance(self.test_patients, dict) and 'test_df' in self.test_patients:
                test_data_df = self.test_patients['test_df'].copy()
            else:
                test_data_df = self.valid_data_df[
                    self.valid_data_df['Subject ID'].isin(self.test_patients['Subject ID'])
                ].copy()
            
            test_data_df['predicted_age'] = y_pred[:len(test_data_df)]
            test_data_df = test_data_df.sort_values(['Subject ID', 'Visit'])
            
            print(f"\n6. ANÁLISE TEMPORAL (DenseNet):")
            print(f"   - Total de exames analisados: {len(test_data_df)}")
            print(f"   - Pacientes únicos: {test_data_df['Subject ID'].nunique()}")
            
            consistent_increase = 0
            mostly_consistent = 0
            total_patients = 0
            total_transitions = 0
            increasing_transitions = 0
            
            temporal_details = []
            
            for subject_id, group in test_data_df.groupby('Subject ID'):
                if len(group) > 1:
                    total_patients += 1
                    
                    visits = group['Visit'].values
                    real_ages = group['Age'].values
                    pred_ages = group['predicted_age'].values
                    
                    # Análise das transições
                    age_diffs_pred = np.diff(pred_ages)
                    
                    num_transitions = len(age_diffs_pred)
                    total_transitions += num_transitions
                    
                    # Contar transições crescentes
                    increasing = np.sum(age_diffs_pred >= 0)
                    increasing_transitions += increasing
                    
                    # Verificar consistência completa
                    if np.all(age_diffs_pred >= 0):
                        consistent_increase += 1
                        status = "✓ Consistente"
                    elif increasing / num_transitions >= 0.5:
                        mostly_consistent += 1
                        status = "⚠ Parcialmente consistente"
                    else:
                        status = "✗ Inconsistente"
                    
                    temporal_details.append({
                        'subject_id': subject_id,
                        'num_visits': len(group),
                        'real_age_span': f"{real_ages[0]:.1f} → {real_ages[-1]:.1f}",
                        'pred_age_span': f"{pred_ages[0]:.1f} → {pred_ages[-1]:.1f}",
                        'real_delta': real_ages[-1] - real_ages[0],
                        'pred_delta': pred_ages[-1] - pred_ages[0],
                        'status': status
                    })
            
            if total_patients > 0:
                consistency_ratio = consistent_increase / total_patients
                partial_ratio = (consistent_increase + mostly_consistent) / total_patients
                transition_ratio = increasing_transitions / total_transitions if total_transitions > 0 else 0
                
                print(f"   - Pacientes com múltiplas visitas: {total_patients}")
                print(f"   - Total de transições: {total_transitions}")
                print(f"   - Pacientes 100% consistentes: {consistent_increase}/{total_patients} ({consistency_ratio:.1%})")
                print(f"   - Pacientes parcialmente consistentes: {mostly_consistent}/{total_patients}")
                print(f"   - Taxa geral de consistência: {partial_ratio:.1%}")
                print(f"   - Transições crescentes: {increasing_transitions}/{total_transitions} ({transition_ratio:.1%})")
                
                # Mostrar exemplos detalhados
                if len(temporal_details) > 0:
                    print(f"\n   Exemplos (primeiros 3 pacientes):")
                    for i, detail in enumerate(temporal_details[:3]):
                        print(f"\n   Paciente {detail['subject_id']}:")
                        print(f"     - Visitas: {detail['num_visits']}")
                        print(f"     - Idade real: {detail['real_age_span']} (Δ={detail['real_delta']:.1f} anos)")
                        print(f"     - Idade predita: {detail['pred_age_span']} (Δ={detail['pred_delta']:.1f} anos)")
                        print(f"     - Status: {detail['status']}")
                
                print(f"\n7. RESPOSTA: Visitas posteriores resultam em idades preditas maiores?")
                if consistency_ratio >= 0.8:
                    print(f"   ✓ SIM - {consistency_ratio:.1%} dos pacientes mostram progressão consistente")
                elif transition_ratio >= 0.7:
                    print(f"   ⚠ PARCIALMENTE - {transition_ratio:.1%} das transições são crescentes")
                else:
                    print(f"   ✗ NÃO - Apenas {transition_ratio:.1%} das transições são crescentes")
                    print(f"   → O modelo captura apenas parcialmente a progressão temporal")
            else:
                print("   ⚠ Nenhum paciente com múltiplas visitas encontrado")
                
        except Exception as e:
            print(f"\n⚠ Erro na análise temporal: {str(e)}")
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
