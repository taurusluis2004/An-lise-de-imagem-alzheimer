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
from sklearn.metrics import confusion_matrix, accuracy_score, recall_score, mean_absolute_error, mean_squared_error, r2_score
import xgboost as xgb
import nibabel as nib
from skimage.transform import resize
from skimage.feature import graycomatrix, graycoprops
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam

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
    """Preparação detalhada dos dados por paciente (split 80/20 + validação).
    Retorna também estrutura 'split_info' com DataFrames e estatísticas.
    """
    base_dir = os.path.dirname(os.path.abspath(__file__))
    csv_path = os.path.join(base_dir, 'oasis_longitudinal_demographic.csv')
    demographics_df = pd.read_csv(csv_path, sep=';')
    demographics_df['CDR'] = demographics_df['CDR'].astype(str).str.replace(',', '.', regex=False)
    demographics_df['CDR'] = pd.to_numeric(demographics_df['CDR'], errors='coerce')
    demographics_df['Age'] = pd.to_numeric(demographics_df['Age'], errors='coerce')

    def get_diagnosis_binary(row):
        if row['Group'] == 'Nondemented':
            return 'NonDemented'
        if row['Group'] == 'Demented':
            return 'Demented'
        if row['Group'] == 'Converted':
            return 'Demented' if row['CDR'] > 0 else 'NonDemented'
        return 'Unknown'

    demographics_df['Diagnosis'] = demographics_df.apply(get_diagnosis_binary, axis=1)
    valid_data_df = demographics_df[demographics_df['Diagnosis'] != 'Unknown'].dropna(subset=['MRI ID','Subject ID','Age']).copy()

    patient_info = valid_data_df.groupby('Subject ID').agg(
        final_CDR=('CDR','max'),
        first_visit_age=('Age','min'),
        num_visits=('Visit','count')
    ).reset_index()
    patient_info['Diagnosis'] = patient_info['final_CDR'].apply(lambda cdr: 'Demented' if cdr>0 else 'NonDemented')
    le = LabelEncoder()
    patient_info['Diagnosis_encoded'] = le.fit_transform(patient_info['Diagnosis'])

    train_val_patients, test_patients = train_test_split(
        patient_info, test_size=0.2, random_state=42, stratify=patient_info['Diagnosis_encoded']
    )
    train_patients, val_patients = train_test_split(
        train_val_patients, test_size=0.2, random_state=42, stratify=train_val_patients['Diagnosis_encoded']
    )

    def get_set_data(p_df):
        set_df = valid_data_df[valid_data_df['Subject ID'].isin(p_df['Subject ID'])].copy()
        axl_dir = os.path.join(base_dir, 'axl')
        paths = [os.path.join(axl_dir, f"{mri_id}_axl.nii") for mri_id in set_df['MRI ID']]
        class_labels = le.transform(set_df['Diagnosis'])
        age_labels = set_df['Age'].values
        return paths, class_labels, age_labels, set_df

    train_paths, y_train_class, y_train_age, train_df = get_set_data(train_patients)
    val_paths, y_val_class, y_val_age, val_df = get_set_data(val_patients)
    test_paths, y_test_class, y_test_age, test_df = get_set_data(test_patients)

    x_train_img = load_images(train_paths)
    x_val_img = load_images(val_paths)
    x_test_img = load_images(test_paths)

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
        'num_test_exams': len(test_paths)
    }

    return (x_train_img, y_train_class, y_train_age), \
           (x_val_img, y_val_class, y_val_age), \
           (x_test_img, y_test_class, y_test_age), \
           valid_data_df, split_info

def extract_features(images):
    """Extrai features de textura multi-distâncias/ângulos + estatísticas + quadrantes."""
    features = []
    distances = [1,3,5]
    angles = [0, np.pi/4, np.pi/2, 3*np.pi/4]
    for img in images:
        img_8bit = (img * 255).astype(np.uint8)
        vec = []
        for d in distances:
            glcm = graycomatrix(img_8bit, distances=[d], angles=angles, levels=256, symmetric=True, normed=True)
            for prop in ['contrast','dissimilarity','homogeneity','energy','correlation','ASM']:
                vals = graycoprops(glcm, prop)[0,:]
                vec.append(vals.mean())
                vec.append(vals.std())
        # estatísticas globais
        vec.extend([
            np.mean(img), np.std(img), np.median(img), np.min(img), np.max(img),
            np.percentile(img,25), np.percentile(img,75), np.var(img)
        ])
        h,w = img.shape
        quads = [img[:h//2,:w//2], img[:h//2,w//2:], img[h//2:,:w//2], img[h//2:,w//2:]]
        for q in quads:
            vec.append(np.mean(q)); vec.append(np.std(q))
        features.append(vec)
    return np.array(features, dtype=np.float32)

def extract_clinical_features(data_df, mri_ids):
    """Extrai features clínicas (EDUC, MMSE, eTIV, nWBV, ASF, Visit, Years_since_first, CDR)."""
    clinical = []
    # calcular primeira visita por paciente (MR Delay mínima)
    first_delay = {}
    for _, row in data_df.iterrows():
        sid = row['Subject ID']
        delay = row.get('MR Delay', 0) if not pd.isna(row.get('MR Delay', 0)) else 0
        if sid not in first_delay or delay < first_delay[sid]:
            first_delay[sid] = delay
    def sf(v, default):
        if pd.isna(v): return default
        if isinstance(v,str):
            v = v.replace(',','.')
        try:
            return float(v)
        except:
            return default
    for mri in mri_ids:
        row = data_df[data_df['MRI ID']==mri]
        if len(row)==0:
            clinical.append([12.0,25.0,1500.0,0.7,1.2,1.0,0.0,0.0]); continue
        r = row.iloc[0]
        sid = r['Subject ID']
        delay = sf(r.get('MR Delay',0),0)
        years_since = max(0, delay - first_delay.get(sid,0))
        clinical.append([
            sf(r.get('EDUC',12.0),12.0),
            sf(r.get('MMSE',25.0),25.0),
            sf(r.get('eTIV',1500.0),1500.0),
            sf(r.get('nWBV',0.7),0.7),
            sf(r.get('ASF',1.2),1.2),
            sf(r.get('Visit',1.0),1.0),
            years_since,
            sf(r.get('CDR',0.0),0.0)
        ])
    return np.array(clinical, dtype=np.float32)

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
        self.densenet_history = None
        
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
        menu_densenet.add_command(label="Curvas Classificação", command=self.plot_curvas_densenet_classif)
        menu_bar.add_cascade(label="DenseNet", menu=menu_densenet)
        
        # Menu Acessibilidade
        menu_acess = tk.Menu(menu_bar, tearoff=0)
        menu_acess.add_command(label="Aumentar Fonte", command=self.aumentar_fonte)
        menu_acess.add_command(label="Diminuir Fonte", command=self.diminuir_fonte)
        menu_bar.add_cascade(label="Acessibilidade", menu=menu_acess)

        # Menu Segmentação
        menu_seg = tk.Menu(menu_bar, tearoff=0)
        menu_seg.add_command(label="Segmentar Ventrículos", command=self.segmentar_ventriculos)
        menu_seg.add_command(label="Mostrar Todas Fatias", command=self.exibir_todas_fatias)
        menu_bar.add_cascade(label="Segmentação", menu=menu_seg)
        
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
        
        # === PAINEL DIREITO COM TABS ===
        self.tabs = ttk.Notebook(self.painel_direito)
        self.tab_imagem = ttk.Frame(self.tabs)
        self.tab_graficos = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_imagem, text="Imagem")
        self.tabs.add(self.tab_graficos, text="Gráficos")
        self.tabs.pack(fill=tk.BOTH, expand=True)

        # Figura principal (Imagem)
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Visualizador de Imagens Médicas")
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_imagem)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.text(0.5, 0.5, 'Carregue uma imagem\npelo menu Arquivo',
                 ha='center', va='center', fontsize=16, transform=self.ax.transAxes)

        # Figura gráficos
        self.fig_plots = Figure(figsize=(8, 6), dpi=100)
        self.ax_plots = self.fig_plots.add_subplot(111)
        self.ax_plots.axis('off')
        self.canvas_plots = FigureCanvasTkAgg(self.fig_plots, master=self.tab_graficos)
        self.canvas_plots.draw()
        self.canvas_plots.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_plots.text(0.5, 0.5, 'Gráficos aparecerão aqui', ha='center', va='center', fontsize=14, transform=self.ax_plots.transAxes)

    # =========================================================================
    # UTILITÁRIOS DE FIGURA
    # =========================================================================
    def show_figure(self, fig, title="Figura"):
        """Exibe figura matplotlib em janela separada Tkinter."""
        win = tk.Toplevel(self.root)
        win.title(title)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        btn = ttk.Button(win, text="Fechar", command=win.destroy)
        btn.pack(pady=5)

    def show_plot_main(self, fig):
        """Renderiza figura na aba 'Gráficos'."""
        # Limpa figura anterior
        self.fig_plots.clf()
        # Copia axes da figura recebida
        # Estratégia: desenhar a figura recebida diretamente no canvas da aba usando Agg.
        # Mais simples: substituir self.fig_plots por fig e recriar canvas.
        self.fig_plots = fig
        # Recria canvas
        for child in self.tab_graficos.winfo_children():
            child.destroy()
        self.canvas_plots = FigureCanvasTkAgg(self.fig_plots, master=self.tab_graficos)
        self.canvas_plots.draw()
        self.canvas_plots.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tabs.select(self.tab_graficos)
    
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
            self.train_data, self.val_data, self.test_data, self.valid_data_df, self.split_info = load_and_prepare_data()
            messagebox.showinfo("Sucesso", f"Dados preparados!\n\n"
                              f"Exames Treino: {self.split_info['num_train_exams']}\n"
                              f"Exames Validação: {self.split_info['num_val_exams']}\n"
                              f"Exames Teste: {self.split_info['num_test_exams']}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao preparar dados:\n{str(e)}")
    
    def extrair_caracteristicas(self):
        """Extrai características GLCM das imagens"""
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        try:
            messagebox.showinfo("Extraindo", "Extraindo características...\nIsso pode levar alguns minutos.")
            # Textura
            tex_train = extract_features(self.train_data[0])
            tex_val = extract_features(self.val_data[0])
            tex_test = extract_features(self.test_data[0])
            # Clínicas (usa DataFrames do split_info)
            train_mri_ids = self.split_info['train_df']['MRI ID'].tolist()
            val_mri_ids = self.split_info['val_df']['MRI ID'].tolist()
            test_mri_ids = self.split_info['test_df']['MRI ID'].tolist()
            clin_train = extract_clinical_features(self.valid_data_df, train_mri_ids)
            clin_val = extract_clinical_features(self.valid_data_df, val_mri_ids)
            clin_test = extract_clinical_features(self.valid_data_df, test_mri_ids)
            # Concatena (textura + clínica)
            self.x_train_features = np.concatenate([tex_train, clin_train], axis=1)
            self.x_val_features = np.concatenate([tex_val, clin_val], axis=1)
            self.x_test_features = np.concatenate([tex_test, clin_test], axis=1)
            messagebox.showinfo("Sucesso", f"Características extraídas!\nDimensão final: {self.x_train_features.shape[1]}")
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
            
            fig = Figure(figsize=(6,5), dpi=100)
            ax = fig.add_subplot(111)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
            ax.set_title('Matriz de Confusão - SVM')
            ax.set_ylabel('Real')
            ax.set_xlabel('Predito')
            self.show_plot_main(fig)
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
            params = dict(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='mae'  # compatível com versões antigas (no params)
            )
            self.xgboost_model = xgb.XGBRegressor(**params)
            try:
                # Tentativa com early stopping e eval_set (APIs mais novas)
                self.xgboost_model.fit(
                    self.x_train_features,
                    self.train_data[2],
                    eval_set=[(self.x_val_features, self.val_data[2])],
                    early_stopping_rounds=30,
                    verbose=False
                )
            except TypeError:
                # Fallback para APIs antigas sem esses kwargs
                self.xgboost_model.fit(self.x_train_features, self.train_data[2])
            # Recupera melhor iteração de forma robusta (difere por versão do XGBoost)
            best_iter = getattr(self.xgboost_model, 'best_iteration', None)
            if best_iter is None:
                try:
                    best_iter = self.xgboost_model.get_booster().best_iteration
                except Exception:
                    best_iter = None
            if best_iter is not None and isinstance(best_iter, (int, np.integer)) and best_iter >= 0:
                message = f"XGBoost treinado! Melhor iteração: {int(best_iter)}"
            else:
                message = "XGBoost treinado!"
            messagebox.showinfo("Sucesso", message)
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao treinar XGBoost:\n{str(e)}")
    
    def avaliar_xgboost(self):
        """Avaliação avançada XGBoost: MAE, RMSE, R2, dispersão, histograma de erros."""
        if self.xgboost_model is None:
            messagebox.showwarning("Aviso", "Treine o XGBoost primeiro!")
            return
        
        try:
            y_true = self.test_data[2]
            y_pred = self.xgboost_model.predict(self.x_test_features)
            mae = mean_absolute_error(y_true, y_pred)
            # Compatível com versões antigas do scikit-learn (sem parâmetro 'squared')
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            abs_err = np.abs(y_true - y_pred)
            pct_within_5 = np.mean(abs_err <= 5) * 100
            pct_within_10 = np.mean(abs_err <= 10) * 100
            # Figura combinada
            fig = Figure(figsize=(10,4), dpi=100)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.scatter(y_true, y_pred, alpha=0.7)
            min_v, max_v = y_true.min(), y_true.max()
            ax1.plot([min_v, max_v], [min_v, max_v], 'r--')
            ax1.set_title('Dispersão Idade (XGBoost)')
            ax1.set_xlabel('Real'); ax1.set_ylabel('Predita')
            ax2.hist(abs_err, bins=20, color='steelblue', alpha=0.8)
            ax2.set_title('Histograma Erros Absolutos')
            ax2.set_xlabel('Erro (anos)'); ax2.set_ylabel('Frequência')
            fig.tight_layout()
            self.show_plot_main(fig)
            messagebox.showinfo("XGBoost", f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.3f}\n±5 anos: {pct_within_5:.1f}%\n±10 anos: {pct_within_10:.1f}%")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
    
    def analise_temporal(self):
        """Análise temporal avançada: verifica crescimento monotônico das idades preditas por paciente."""
        if self.xgboost_model is None:
            messagebox.showwarning("Aviso", "Treine o XGBoost primeiro!")
            return
        
        try:
            y_pred = self.xgboost_model.predict(self.x_test_features)
            # DataFrame de teste completo
            test_df = self.split_info['test_df'].copy()
            test_df = test_df.sort_values(['Subject ID','Visit'])
            test_df['PredictedAge'] = y_pred[:len(test_df)]
            multi_patients = 0
            consistent = 0
            for sid, grp in test_df.groupby('Subject ID'):
                if len(grp) > 1:
                    multi_patients += 1
                    if np.all(np.diff(grp['PredictedAge'].values) >= -0.5):  # tolera pequena oscilação negativa
                        consistent += 1
            perc = (consistent / multi_patients * 100) if multi_patients>0 else 0
            # Plot linhas por paciente (limitando a até 12 pacientes para legibilidade)
            fig = Figure(figsize=(8,5), dpi=100)
            ax = fig.add_subplot(111)
            for i,(sid, grp) in enumerate(test_df.groupby('Subject ID')):
                if i>=12: break
                ax.plot(grp['Visit'], grp['PredictedAge'], marker='o', label=str(sid))
            ax.set_title('Trajetória Idade Predita (Teste)')
            ax.set_xlabel('Visita'); ax.set_ylabel('Idade Predita')
            if multi_patients>0:
                ax.legend(fontsize=8, ncol=2)
            fig.tight_layout()
            self.show_plot_main(fig)
            messagebox.showinfo("Análise Temporal", f"Pacientes multi-visita: {multi_patients}\nConsistentes crescimento: {consistent}\nPercentual: {perc:.1f}%")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro:\n{str(e)}")
    
    # =========================================================================
    # DENSENET - MODELO PROFUNDO
    # =========================================================================
    def _prepare_images_densenet(self, images):
        """Converte lista de imagens (N,H,W) para (N,224,224,3) normalizadas."""
        target = (224,224)
        out = []
        for img in images:
            arr = img.astype(np.float32)
            mn, mx = arr.min(), arr.max()
            if mx > mn:
                arr = (arr - mn)/(mx-mn)
            else:
                arr = np.zeros_like(arr, dtype=np.float32)
            arr_res = resize(arr, target, anti_aliasing=True).astype(np.float32)
            arr_rgb = np.stack([arr_res, arr_res, arr_res], axis=-1)
            out.append(arr_rgb)
        return np.array(out, dtype=np.float32)

    def treinar_densenet_classif(self):
        """Treina DenseNet121 (ImageNet) para classificação binária."""
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        try:
            x_train, y_train, _ = self.train_data
            x_val, y_val, _ = self.val_data
            messagebox.showinfo("DenseNet", "Preparando imagens para DenseNet...")
            self.x_train_densenet = self._prepare_images_densenet(x_train)
            self.x_val_densenet = self._prepare_images_densenet(x_val)
            base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))
            for layer in base.layers:
                layer.trainable = False
            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            out = Dense(1, activation='sigmoid')(x)
            model = Model(inputs=base.input, outputs=out)
            model.compile(optimizer=Adam(learning_rate=1e-3), loss='binary_crossentropy', metrics=['accuracy'])
            messagebox.showinfo("DenseNet", "Treinando (5 épocas)...")
            history = model.fit(self.x_train_densenet, y_train, validation_data=(self.x_val_densenet, y_val), epochs=5, batch_size=8, verbose=1)
            self.densenet_classif = model
            self.densenet_history = history
            messagebox.showinfo("Sucesso", "DenseNet classificação treinada!")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha no treinamento DenseNet:\n{str(e)}")
    
    def treinar_densenet_regress(self):
        """Treina DenseNet121 para regressão de idade."""
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        try:
            x_train, _, y_train_age = self.train_data
            x_val, _, y_val_age = self.val_data
            messagebox.showinfo("DenseNet", "Preparando imagens para regressão...")
            self.x_train_densenet_reg = self._prepare_images_densenet(x_train)
            self.x_val_densenet_reg = self._prepare_images_densenet(x_val)
            base = DenseNet121(weights='imagenet', include_top=False, input_shape=(224,224,3))
            for layer in base.layers:
                layer.trainable = False
            x = base.output
            x = GlobalAveragePooling2D()(x)
            x = Dense(128, activation='relu')(x)
            x = Dropout(0.3)(x)
            out = Dense(1, activation='linear')(x)
            model = Model(inputs=base.input, outputs=out)
            model.compile(optimizer=Adam(learning_rate=1e-3), loss='mse', metrics=['mae'])
            history = model.fit(self.x_train_densenet_reg, y_train_age, validation_data=(self.x_val_densenet_reg, y_val_age), epochs=5, batch_size=8, verbose=1)
            self.densenet_regress = model
            self.densenet_regress_history = history
            # Curvas
            fig = Figure(figsize=(8,5), dpi=100)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.plot(history.history['mae'], label='Treino')
            ax1.plot(history.history['val_mae'], label='Validação')
            ax1.set_title('MAE')
            ax1.set_xlabel('Época'); ax1.set_ylabel('MAE'); ax1.legend()
            ax2.plot(history.history['loss'], label='Treino')
            ax2.plot(history.history['val_loss'], label='Validação')
            ax2.set_title('MSE')
            ax2.set_xlabel('Época'); ax2.set_ylabel('MSE'); ax2.legend()
            fig.tight_layout()
            self.show_plot_main(fig)
            messagebox.showinfo("Sucesso", "DenseNet regressão treinada!")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha no treinamento DenseNet Regressão:\n{str(e)}")
    
    def plot_curvas_densenet_classif(self):
        """Plota curvas (acurácia e loss) da DenseNet classificação."""
        if self.densenet_history is None:
            messagebox.showwarning("Aviso", "Treine a DenseNet primeiro!")
            return
        h = self.densenet_history.history
        fig = Figure(figsize=(8,5), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(h['accuracy'], label='Treino')
        ax1.plot(h['val_accuracy'], label='Validação')
        ax1.set_title('Acurácia')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Acc')
        ax1.legend()
        ax2.plot(h['loss'], label='Treino')
        ax2.plot(h['val_loss'], label='Validação')
        ax2.set_title('Loss')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Loss')
        ax2.legend()
        fig.tight_layout()
        self.show_figure(fig, title="Curvas DenseNet Classificação")

    def avaliar_densenet_classif(self):
        """Avalia DenseNet classificação: métricas e matriz de confusão."""
        if self.densenet_classif is None or self.test_data is None:
            messagebox.showwarning("Aviso", "Treine DenseNet e prepare dados!")
            return
        try:
            x_test, y_test, _ = self.test_data
            x_test_proc = self._prepare_images_densenet(x_test)
            probs = self.densenet_classif.predict(x_test_proc)
            y_pred = (probs.ravel() >= 0.5).astype(int)
            acc, sens, spec, cm = evaluate_classifier(y_test, y_pred, "DenseNet Classif")
            fig = Figure(figsize=(6,5), dpi=100)
            ax = fig.add_subplot(111)
            sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', ax=ax)
            ax.set_title('Matriz de Confusão - DenseNet')
            ax.set_ylabel('Real')
            ax.set_xlabel('Predito')
            self.show_figure(fig, title="Matriz DenseNet Classificação")
            messagebox.showinfo("Resultados DenseNet", f"Acurácia: {acc:.4f}\nSensibilidade: {sens:.4f}\nEspecificidade: {spec:.4f}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha na avaliação DenseNet:\n{str(e)}")
    
    def avaliar_densenet_regress(self):
        """Avalia DenseNet regressão no teste."""
        if self.densenet_regress is None or self.test_data is None:
            messagebox.showwarning("Aviso", "Treine a DenseNet regressão e prepare dados!")
            return
        try:
            x_test, _, y_test_age = self.test_data
            x_test_proc = self._prepare_images_densenet(x_test)
            preds = self.densenet_regress.predict(x_test_proc).ravel()
            mae = mean_absolute_error(y_test_age, preds)
            rmse = np.sqrt(mean_squared_error(y_test_age, preds))
            r2 = r2_score(y_test_age, preds)
            # Dispersão
            fig = Figure(figsize=(6,5), dpi=100)
            ax = fig.add_subplot(111)
            ax.scatter(y_test_age, preds, alpha=0.7)
            ax.plot([y_test_age.min(), y_test_age.max()], [y_test_age.min(), y_test_age.max()], 'r--')
            ax.set_title('Idade Real vs Predita (DenseNet)')
            ax.set_xlabel('Real'); ax.set_ylabel('Predita')
            self.show_plot_main(fig)
            messagebox.showinfo("DenseNet Regressão", f"MAE: {mae:.2f}\nRMSE: {rmse:.2f}\nR2: {r2:.3f}")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha na avaliação DenseNet Regressão:\n{str(e)}")

    # =========================================================================
    # SEGMENTAÇÃO VENTRÍCULOS (K-Means simples)
    # =========================================================================
    def segmentar_ventriculos(self):
        """Segmenta ventrículos na fatia atual usando K-means (k=2) heurístico."""
        if self.current_image_data is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!")
            return
        try:
            # Usa fatia atual
            if len(self.current_image_data.shape) == 3:
                slice_data = ImageLoader.get_slice(self.current_image_data, self.current_axis, self.current_slice_index)
            else:
                slice_data = self.current_image_data
            try:
                import cv2
            except ImportError:
                messagebox.showerror("Erro", "OpenCV não instalado. Execute: pip install opencv-python")
                return
            img = slice_data.astype(np.float32)
            # Normaliza
            mn, mx = img.min(), img.max()
            if mx > mn:
                img_n = (img - mn)/(mx-mn)
            else:
                img_n = np.zeros_like(img)
            img_blur = cv2.GaussianBlur(img_n, (5,5), 1)
            pixels = img_blur.reshape((-1,1)).astype(np.float32)
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 100, 0.85)
            _, labels, centers = cv2.kmeans(pixels, 2, None, criteria, 10, cv2.KMEANS_RANDOM_CENTERS)
            centers = centers.ravel()
            # Assume ventrículos = cluster mais claro
            ventr_cluster = np.argmax(centers)
            mask = (labels.reshape(img_blur.shape) == ventr_cluster)
            # Limpa máscara
            mask = cv2.medianBlur(mask.astype(np.uint8), 5)
            # Figura
            fig = Figure(figsize=(8,4), dpi=100)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.imshow(img_n, cmap='gray'); ax1.set_title('Fatia Original'); ax1.axis('off')
            overlay = np.dstack([np.zeros_like(img_n), np.zeros_like(img_n), np.zeros_like(img_n)])
            overlay[mask] = [1,0,0]
            ax2.imshow(img_n, cmap='gray')
            ax2.imshow(overlay, alpha=0.4)
            ax2.set_title('Ventrículos (Heurístico)'); ax2.axis('off')
            fig.tight_layout()
            self.show_plot_main(fig)
        except Exception as e:
            messagebox.showerror("Erro", f"Falha na segmentação:\n{str(e)}")

    def exibir_todas_fatias(self):
        """Exibe grid com todas as fatias (se 3D)."""
        if self.current_image_data is None or len(self.current_image_data.shape) != 3:
            messagebox.showwarning("Aviso", "Imagem 3D não carregada.")
            return
        data = self.current_image_data
        num_slices = data.shape[2]
        cols = 8
        rows = int(np.ceil(num_slices/cols))
        fig = Figure(figsize=(15, rows*2), dpi=100)
        for i in range(num_slices):
            ax = fig.add_subplot(rows, cols, i+1)
            ax.imshow(ImageLoader.normalize_for_display(data[:,:,i]), cmap='gray')
            ax.axis('off')
        fig.tight_layout()
        self.show_plot_main(fig)
    
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
