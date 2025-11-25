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
from skimage.measure import label, regionprops
import cv2
from PIL import Image
import tkinter as tk
from tkinter import ttk, messagebox, filedialog
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Dropout, GlobalAveragePooling2D
from tensorflow.keras.applications import DenseNet121
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import Callback

class TestAccuracyCallback(Callback):
    def __init__(self, x_test, y_test):
        super().__init__()
        self.x_test = x_test
        self.y_test = y_test
        self.test_acc = []

    def on_epoch_end(self, epoch, logs=None):
        loss, acc = self.model.evaluate(self.x_test, self.y_test, verbose=0)
        if logs is not None:
            logs['test_accuracy'] = acc
        self.test_acc.append(acc)
        print(f"Época {epoch+1} - Test Accuracy: {acc:.4f}")

class ImageLoader:
    
    @staticmethod
    def load_image(file_path):
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
    def normalize_for_display(image_data):
        data = image_data.copy()
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        data_min = data.min()
        data_max = data.max()
        
        if data_max > data_min:
            data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
        
        return data

image_size = (128, 128)

def load_images(paths):
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

    planilha_path = os.path.join(base_dir, 'planilha.csv')
    planilha_df = None
    if os.path.exists(planilha_path):
        planilha_df = pd.read_csv(planilha_path)
        for col in ['nWBV', 'area', 'perimeter', 'circularity', 'eccentricity', 'solidity', 'extent', 'mean_intensity']:
            if col in planilha_df.columns:
                planilha_df[col] = planilha_df[col].astype(str).str.replace(',', '.', regex=False)
                planilha_df[col] = pd.to_numeric(planilha_df[col], errors='coerce')

    split_info = {
        'train_patients': train_patients,
        'val_patients': val_patients,
        'test_patients': test_patients,
        'train_df': train_df,
        'val_df': val_df,
        'test_df': test_df,
        'planilha_df': planilha_df,
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

def extract_morphological_features(planilha_df, mri_ids):
    if planilha_df is None:
        return np.zeros((len(mri_ids), 7), dtype=np.float32)
    
    morphological = []
    for mri in mri_ids:
        row = planilha_df[planilha_df['MRI ID'] == mri]
        if len(row) == 0:
            morphological.append([1000.0, 300.0, 0.15, 0.7, 0.6, 0.4, 45.0])
            continue
        r = row.iloc[0]
        morphological.append([
            float(r.get('area', 1000.0)) if not pd.isna(r.get('area')) else 1000.0,
            float(r.get('perimeter', 300.0)) if not pd.isna(r.get('perimeter')) else 300.0,
            float(r.get('circularity', 0.15)) if not pd.isna(r.get('circularity')) else 0.15,
            float(r.get('eccentricity', 0.7)) if not pd.isna(r.get('eccentricity')) else 0.7,
            float(r.get('solidity', 0.6)) if not pd.isna(r.get('solidity')) else 0.6,
            float(r.get('extent', 0.4)) if not pd.isna(r.get('extent')) else 0.4,
            float(r.get('mean_intensity', 45.0)) if not pd.isna(r.get('mean_intensity')) else 45.0
        ])
    return np.array(morphological, dtype=np.float32)

def extract_clinical_features(data_df, mri_ids):
    clinical = []
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
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title(f'{model_name} Learning Curve')
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy')
    plt.legend(loc='lower right')
    plt.show()

class AlzheimerAnalysisGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Análise de Imagens - Alzheimer")
        self.root.geometry("1200x800")
        
        self.current_image_data = None
        self.current_metadata = None
        self.zoom_level = 1.0
        
        self.dataset_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'axl')
        
        self.svm_model = None
        self.xgboost_model = None
        self.densenet_classif = None
        self.densenet_regress = None
        self.densenet_history = None
        
        self.train_data = None
        self.val_data = None
        self.test_data = None
        self.valid_data_df = None
        self.test_patients = None
        
        self.x_train_features = None
        self.x_val_features = None
        self.x_test_features = None
        
        self.font_size = 12
        self.base_font = ("Arial", self.font_size)
        
        self.setup_ui()
    
    def setup_ui(self):
        self.style = ttk.Style()
        self.style.configure("TButton", font=self.base_font, padding=5)
        self.style.configure("TLabel", font=self.base_font)
        
        menu_bar = tk.Menu(self.root)
        
        menu_arquivo = tk.Menu(menu_bar, tearoff=0)
        menu_arquivo.add_command(label="Carregar Imagem", command=self.carregar_dataset)
        menu_arquivo.add_command(label="Carregar Arquivo Externo", command=self.carregar_arquivo_externo)
        menu_arquivo.add_separator()
        menu_arquivo.add_command(label="Sair", command=self.root.quit)
        menu_bar.add_cascade(label="Arquivo", menu=menu_arquivo)
        
        menu_dados = tk.Menu(menu_bar, tearoff=0)
        menu_dados.add_command(label="Preparar Dados (80/20)", command=self.preparar_dados)
        menu_dados.add_separator()
        menu_dados.add_command(label="Extrair Características (Classificação)", command=self.extrair_caracteristicas)
        menu_dados.add_command(label="Extrair Características (Regressão)", command=self.extrair_caracteristicas_regressao)
        menu_bar.add_cascade(label="Dados", menu=menu_dados)
        
        menu_svm = tk.Menu(menu_bar, tearoff=0)
        menu_svm.add_command(label="Treinar SVM", command=self.treinar_svm)
        menu_svm.add_command(label="Avaliar SVM", command=self.avaliar_svm)
        menu_svm.add_command(label="Matriz de Confusão", command=self.matriz_confusao_svm)
        menu_bar.add_cascade(label="SVM", menu=menu_svm)
        
        menu_xgb = tk.Menu(menu_bar, tearoff=0)
        menu_xgb.add_command(label="Treinar XGBoost", command=self.treinar_xgboost)
        menu_xgb.add_command(label="Avaliar XGBoost", command=self.avaliar_xgboost)
        menu_xgb.add_command(label="Análise Temporal", command=self.analise_temporal)
        menu_bar.add_cascade(label="XGBoost", menu=menu_xgb)
        
        menu_densenet = tk.Menu(menu_bar, tearoff=0)
        menu_densenet.add_command(label="Treinar Classificação", command=self.treinar_densenet_classif)
        menu_densenet.add_command(label="Treinar Regressão", command=self.treinar_densenet_regress)
        menu_densenet.add_separator()
        menu_densenet.add_command(label="Avaliar Classificação", command=self.avaliar_densenet_classif)
        menu_densenet.add_command(label="Avaliar Regressão", command=self.avaliar_densenet_regress)
        menu_densenet.add_command(label="Curvas Classificação", command=self.plot_curvas_densenet_classif)
        menu_bar.add_cascade(label="DenseNet", menu=menu_densenet)

        menu_vis = tk.Menu(menu_bar, tearoff=0)
        menu_vis.add_command(label="Gráficos de Dispersão (planilha)", command=self.plot_scatter_matrix)
        menu_bar.add_cascade(label="Visualização", menu=menu_vis)
        
        menu_acess = tk.Menu(menu_bar, tearoff=0)
        menu_acess.add_command(label="Aumentar Fonte", command=self.aumentar_fonte)
        menu_acess.add_command(label="Diminuir Fonte", command=self.diminuir_fonte)
        menu_bar.add_cascade(label="Acessibilidade", menu=menu_acess)

        menu_seg = tk.Menu(menu_bar, tearoff=0)
        menu_seg.add_command(label="Segmentar Ventrículos", command=self.segmentar_ventriculos)
        menu_bar.add_cascade(label="Segmentação", menu=menu_seg)
        
        self.root.config(menu=menu_bar)
        
        self.painel_esquerdo = ttk.Frame(self.root, width=300)
        self.painel_esquerdo.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        self.painel_direito = ttk.Frame(self.root)
        self.painel_direito.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)
        
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
        
        self.zoom_frame = ttk.LabelFrame(self.painel_esquerdo, text="Zoom", padding=10)
        self.zoom_frame.pack(fill=tk.X, pady=10)
        
        btn_frame = ttk.Frame(self.zoom_frame)
        btn_frame.pack()
        
        ttk.Button(btn_frame, text="Zoom In (+)", command=self.zoom_in).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Zoom Out (-)", command=self.zoom_out).pack(side=tk.LEFT, padx=2)
        ttk.Button(btn_frame, text="Reset", command=self.zoom_reset).pack(side=tk.LEFT, padx=2)
        
        self.lbl_zoom = ttk.Label(self.zoom_frame, text="Zoom: 100%")
        self.lbl_zoom.pack(pady=5)
        
        self.tabs = ttk.Notebook(self.painel_direito)
        self.tab_imagem = ttk.Frame(self.tabs)
        self.tab_graficos = ttk.Frame(self.tabs)
        self.tabs.add(self.tab_imagem, text="Imagem")
        self.tabs.add(self.tab_graficos, text="Gráficos")
        self.tabs.pack(fill=tk.BOTH, expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Visualizador de Imagens Médicas")
        self.ax.axis('off')
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.tab_imagem)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax.text(0.5, 0.5, 'Carregue uma imagem\npelo menu Arquivo',
                 ha='center', va='center', fontsize=16, transform=self.ax.transAxes)

        self.fig_plots = Figure(figsize=(8, 6), dpi=100)
        self.ax_plots = self.fig_plots.add_subplot(111)
        self.ax_plots.axis('off')
        self.canvas_plots = FigureCanvasTkAgg(self.fig_plots, master=self.tab_graficos)
        self.canvas_plots.draw()
        self.canvas_plots.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.ax_plots.text(0.5, 0.5, 'Gráficos aparecerão aqui', ha='center', va='center', fontsize=14, transform=self.ax_plots.transAxes)
    
    def show_figure(self, fig, title="Figura"):
        win = tk.Toplevel(self.root)
        win.title(title)
        canvas = FigureCanvasTkAgg(fig, master=win)
        canvas.draw()
        canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        btn = ttk.Button(win, text="Fechar", command=win.destroy)
        btn.pack(pady=5)

    def show_plot_main(self, fig):
        self.fig_plots.clf()
        self.fig_plots = fig
        for child in self.tab_graficos.winfo_children():
            child.destroy()
        self.canvas_plots = FigureCanvasTkAgg(self.fig_plots, master=self.tab_graficos)
        self.canvas_plots.draw()
        self.canvas_plots.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        self.tabs.select(self.tab_graficos)
    
    def carregar_dataset(self):
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
        filetypes = [
            ("Todos os suportados", "*.nii *.nii.gz *.png *.jpg *.jpeg"),
            ("NIfTI", "*.nii *.nii.gz"),
            ("Imagens", "*.png *.jpg *.jpeg")
        ]
        filepath = filedialog.askopenfilename(title="Selecionar Imagem", filetypes=filetypes)
        if filepath:
            self.carregar_e_exibir(filepath)
    
    def carregar_e_exibir(self, filepath):
        try:
            self.current_image_data, self.current_metadata = ImageLoader.load_image(filepath)
            
            self.lbl_arquivo.config(text=f"Arquivo: {os.path.basename(filepath)}")
            self.lbl_formato.config(text=f"Formato: {self.current_metadata['format']}")
            self.lbl_dimensoes.config(text=f"Dimensões: {self.current_metadata['shape']}")
            
            self.zoom_level = 1.0
            self.exibir_imagem()
            
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao carregar imagem:\n{str(e)}")
    
    def exibir_imagem(self):
        if self.current_image_data is None:
            return
        
        self.ax.clear()
        
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
    
    def preparar_dados(self):
        try:
            messagebox.showinfo("Preparando Dados", "Carregando e preparando dados...\nIsso pode levar alguns minutos.")
            self.train_data, self.val_data, self.test_data, self.valid_data_df, self.split_info = load_and_prepare_data()
            messagebox.showinfo("Sucesso", f"Dados preparados!\n\nExames Treino: {self.split_info['num_train_exams']}\nExames Validação: {self.split_info['num_val_exams']}\nExames Teste: {self.split_info['num_test_exams']}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao preparar dados:\n{str(e)}")
    
    def extrair_caracteristicas(self):
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        try:
            messagebox.showinfo("Extraindo", "Extraindo características para CLASSIFICAÇÃO...\nIsso pode levar alguns minutos.")
            tex_train = extract_features(self.train_data[0])
            tex_val = extract_features(self.val_data[0])
            tex_test = extract_features(self.test_data[0])
            train_mri_ids = self.split_info['train_df']['MRI ID'].tolist()
            val_mri_ids = self.split_info['val_df']['MRI ID'].tolist()
            test_mri_ids = self.split_info['test_df']['MRI ID'].tolist()
            clin_train = extract_clinical_features(self.valid_data_df, train_mri_ids)
            clin_val = extract_clinical_features(self.valid_data_df, val_mri_ids)
            clin_test = extract_clinical_features(self.valid_data_df, test_mri_ids)
            self.x_train_features = np.concatenate([tex_train, clin_train], axis=1)
            self.x_val_features = np.concatenate([tex_val, clin_val], axis=1)
            self.x_test_features = np.concatenate([tex_test, clin_test], axis=1)
            messagebox.showinfo("Sucesso", f"Características de CLASSIFICAÇÃO extraídas!\nDimensão: {self.x_train_features.shape[1]} (textura+clínicas)")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao extrair características:\n{str(e)}")
    
    def extrair_caracteristicas_regressao(self):
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        
        try:
            messagebox.showinfo("Extraindo", "Extraindo características para REGRESSÃO...\nIsso pode levar alguns minutos.")
            tex_train = extract_features(self.train_data[0])
            tex_val = extract_features(self.val_data[0])
            tex_test = extract_features(self.test_data[0])
            
            train_mri_ids = self.split_info['train_df']['MRI ID'].tolist()
            val_mri_ids = self.split_info['val_df']['MRI ID'].tolist()
            test_mri_ids = self.split_info['test_df']['MRI ID'].tolist()
            
            planilha_df = self.split_info.get('planilha_df')
            morph_train = extract_morphological_features(planilha_df, train_mri_ids)
            morph_val = extract_morphological_features(planilha_df, val_mri_ids)
            morph_test = extract_morphological_features(planilha_df, test_mri_ids)
            
            clin_train = extract_clinical_features(self.valid_data_df, train_mri_ids)
            clin_val = extract_clinical_features(self.valid_data_df, val_mri_ids)
            clin_test = extract_clinical_features(self.valid_data_df, test_mri_ids)
            
            self.x_train_features_reg = np.concatenate([morph_train, tex_train, clin_train], axis=1)
            self.x_val_features_reg = np.concatenate([morph_val, tex_val, clin_val], axis=1)
            self.x_test_features_reg = np.concatenate([morph_test, tex_test, clin_test], axis=1)
            
            messagebox.showinfo("Sucesso", 
                f"Características de REGRESSÃO extraídas!\n" 
                f"Dimensão: {self.x_train_features_reg.shape[1]} features\n" 
                f"  - Morfológicas (planilha.csv): 7\n" 
                f"  - Textura (GLCM): {tex_train.shape[1]}\n" 
                f"  - Clínicas (OASIS): {clin_train.shape[1]}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao extrair características de regressão:\n{str(e)}")
    
    def treinar_svm(self):
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
        if self.svm_model is None:
            messagebox.showwarning("Aviso", "Treine o SVM primeiro!")
            return
        
        try:
            X_test_scaled = self.svm_scaler.transform(self.x_test_features)
            y_pred = self.svm_model.predict(X_test_scaled)
            acc, sens, spec, cm = evaluate_classifier(self.test_data[1], y_pred, "SVM")
            
            messagebox.showinfo("Resultados SVM", f"Acurácia: {acc:.4f}\nSensibilidade: {sens:.4f}\nEspecificidade: {spec:.4f}")
        except Exception as e:
            messagebox.showerror("Erro", f"Erro ao avaliar SVM:\n{str(e)}")
    
    def matriz_confusao_svm(self):
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
    
    def treinar_xgboost(self):
        if not hasattr(self, 'x_train_features_reg') or self.x_train_features_reg is None:
            messagebox.showwarning("Aviso", "Extraia as características de REGRESSÃO primeiro!\n(Use 'Extrair Características (Regressão)')")
            return
        
        try:
            messagebox.showinfo("Treinando", "Treinando XGBoost com features morfológicas da planilha.csv...")
            params = dict(
                n_estimators=400,
                max_depth=6,
                learning_rate=0.05,
                subsample=0.8,
                colsample_bytree=0.8,
                reg_alpha=0.1,
                reg_lambda=1.0,
                random_state=42,
                eval_metric='mae'
            )
            self.xgboost_model = xgb.XGBRegressor(**params)
            try:
                self.xgboost_model.fit(
                    self.x_train_features_reg,
                    self.train_data[2],
                    eval_set=[(self.x_val_features_reg, self.val_data[2])],
                    early_stopping_rounds=30,
                    verbose=False
                )
            except TypeError:
                self.xgboost_model.fit(self.x_train_features_reg, self.train_data[2])
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
        if self.xgboost_model is None:
            messagebox.showwarning("Aviso", "Treine o XGBoost primeiro!")
            return
        
        try:
            y_true = self.test_data[2]
            y_pred = self.xgboost_model.predict(self.x_test_features_reg)
            mae = mean_absolute_error(y_true, y_pred)
            rmse = np.sqrt(mean_squared_error(y_true, y_pred))
            r2 = r2_score(y_true, y_pred)
            abs_err = np.abs(y_true - y_pred)
            pct_within_5 = np.mean(abs_err <= 5) * 100
            pct_within_10 = np.mean(abs_err <= 10) * 100
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
        if self.xgboost_model is None:
            messagebox.showwarning("Aviso", "Treine o XGBoost primeiro!")
            return
        
        try:
            y_pred = self.xgboost_model.predict(self.x_test_features_reg)
            test_df = self.split_info['test_df'].copy()
            test_df = test_df.sort_values(['Subject ID','Visit'])
            test_df['PredictedAge'] = y_pred[:len(test_df)]
            multi_patients = 0
            consistent = 0
            for sid, grp in test_df.groupby('Subject ID'):
                if len(grp) > 1:
                    multi_patients += 1
                    if np.all(np.diff(grp['PredictedAge'].values) >= -0.5):
                        consistent += 1
            perc = (consistent / multi_patients * 100) if multi_patients>0 else 0
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
    
    def _prepare_images_densenet(self, images):
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
        if self.train_data is None:
            messagebox.showwarning("Aviso", "Prepare os dados primeiro!")
            return
        try:
            x_train, y_train, _ = self.train_data
            x_val, y_val, _ = self.val_data
            x_test, y_test, _ = self.test_data
            messagebox.showinfo("DenseNet", "Preparando imagens para DenseNet...")
            self.x_train_densenet = self._prepare_images_densenet(x_train)
            self.x_val_densenet = self._prepare_images_densenet(x_val)
            self.x_test_densenet = self._prepare_images_densenet(x_test)

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

            cb_test = TestAccuracyCallback(self.x_test_densenet, y_test)
            messagebox.showinfo("DenseNet", "Treinando fase 1 (camadas congeladas, 5 épocas)...")
            history_phase1 = model.fit(
                self.x_train_densenet, y_train,
                validation_data=(self.x_val_densenet, y_val),
                epochs=5, batch_size=8, verbose=1,
                callbacks=[cb_test]
            )

            ft_layers = 50
            for layer in base.layers[-ft_layers:]:
                layer.trainable = True
            model.compile(optimizer=Adam(learning_rate=1e-4), loss='binary_crossentropy', metrics=['accuracy'])
            messagebox.showinfo("DenseNet", f"Fine-tuning (últimas {ft_layers} camadas, 3 épocas)...")
            history_phase2 = model.fit(
                self.x_train_densenet, y_train,
                validation_data=(self.x_val_densenet, y_val),
                epochs=3, batch_size=8, verbose=1,
                callbacks=[cb_test]
            )

            merged_history = {}
            for k in history_phase1.history.keys():
                merged_history[k] = history_phase1.history[k] + history_phase2.history.get(k, [])
            merged_history['test_accuracy'] = cb_test.test_acc

            self.densenet_classif = model
            class SimpleHistory:
                def __init__(self, hist):
                    self.history = hist
            self.densenet_history = SimpleHistory(merged_history)

            fig = Figure(figsize=(10,5), dpi=100)
            ax1 = fig.add_subplot(121)
            ax2 = fig.add_subplot(122)
            ax1.plot(merged_history['accuracy'], label='Treino')
            ax1.plot(merged_history['val_accuracy'], label='Validação')
            ax1.plot(merged_history['test_accuracy'], label='Teste', linestyle='--')
            ax1.set_title('Acurácia por Época')
            ax1.set_xlabel('Época'); ax1.set_ylabel('Accuracy')
            ax1.legend()
            ax2.plot(merged_history['loss'], label='Treino')
            ax2.plot(merged_history['val_loss'], label='Validação')
            ax2.set_title('Loss por Época')
            ax2.set_xlabel('Época'); ax2.set_ylabel('Loss')
            ax2.legend()
            fig.tight_layout()
            self.show_plot_main(fig)

            messagebox.showinfo("Sucesso", "DenseNet classificação (fine-tuning) treinada!")
        except Exception as e:
            messagebox.showerror("Erro", f"Falha no treinamento DenseNet:\n{str(e)}")
    
    def treinar_densenet_regress(self):
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
        if self.densenet_history is None:
            messagebox.showwarning("Aviso", "Treine a DenseNet primeiro!")
            return
        h = self.densenet_history.history
        fig = Figure(figsize=(10,5), dpi=100)
        ax1 = fig.add_subplot(121)
        ax2 = fig.add_subplot(122)
        ax1.plot(h['accuracy'], label='Treino')
        ax1.plot(h['val_accuracy'], label='Validação')
        if 'test_accuracy' in h:
            ax1.plot(h['test_accuracy'], label='Teste', linestyle='--')
        ax1.set_title('Acurácia por Época')
        ax1.set_xlabel('Época')
        ax1.set_ylabel('Accuracy')
        ax1.legend()
        ax2.plot(h['loss'], label='Treino')
        ax2.plot(h['val_loss'], label='Validação')
        ax2.set_title('Loss por Época')
        ax2.set_xlabel('Época')
        ax2.set_ylabel('Loss')
        ax2.legend()
        fig.tight_layout()
        self.show_figure(fig, title="Curvas DenseNet Classificação")

    def avaliar_densenet_classif(self):
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

    def plot_scatter_matrix(self):
        try:
            messagebox.showinfo("Carregando", "Carregando dados da planilha e gerando gráficos...\nIsso pode levar um momento.")
            
            base_dir = os.path.dirname(os.path.abspath(__file__))
            csv_path = os.path.join(base_dir, 'planilha.csv')

            if not os.path.exists(csv_path):
                messagebox.showerror("Erro", f"Arquivo 'planilha.csv' não encontrado no diretório:\n{base_dir}")
                return

            df = pd.read_csv(csv_path)

            feature_cols = ['area', 'perimeter', 'circularity', 'eccentricity', 'solidity', 'extent', 'mean_intensity']
            
            for col in feature_cols:
                if df[col].dtype == 'object':
                    df[col] = df[col].str.replace(',', '.', regex=False)
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            df.dropna(subset=feature_cols, inplace=True)

            if df.empty:
                messagebox.showwarning("Aviso", "Não há dados válidos na planilha para gerar os gráficos.")
                return

            color_palette = {
                'Nondemented': 'blue',
                'Demented': 'red',
                'Converted': 'black'
            }
            
            g = sns.pairplot(df, vars=feature_cols, hue='Group', palette=color_palette, plot_kws={'alpha': 0.6}, diag_kind=None)
            
            g.fig.suptitle("Gráficos de Dispersão por Classe", y=1.02)
            
            self.show_plot_main(g.fig)

            messagebox.showinfo("Sucesso", "Gráficos de dispersão gerados na aba 'Gráficos'.")

        except Exception as e:
            messagebox.showerror("Erro", f"Ocorreu um erro ao gerar os gráficos:\n{str(e)}")
    
    def identify_ventricles(self, x_train_img, idx=0, debug_plots=True):
        if x_train_img.ndim == 3:
            img = x_train_img[idx].astype(np.float32)
        else:
            img = x_train_img.astype(np.float32)

        h, w = img.shape

        img_norm = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        if debug_plots:
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(img_norm, cmap='gray')
            ax.set_title("Original slice (normalized)")
            ax.axis("off")
            self.show_plot_main(fig)

        img_denoised = cv2.GaussianBlur(img_norm, (5, 5), sigmaX=1)

        if debug_plots:
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(img_denoised, cmap='gray')
            ax.set_title("After Gaussian blur")
            ax.axis("off")
            self.show_plot_main(fig)

        pixels = img_denoised.reshape((-1, 1)).astype(np.float32)

        criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            100,
            0.85
        )

        k = 2
        compactness, labels, centers = cv2.kmeans(
            pixels,
            k,
            None,
            criteria,
            10,
            cv2.KMEANS_RANDOM_CENTERS
        )

        centers = centers.flatten()
        dark_cluster_id = np.argmin(centers)

        label_img = labels.reshape((h, w))

        dark_mask = (label_img == dark_cluster_id).astype(np.uint8)

        if debug_plots:
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(dark_mask, cmap='gray')
            ax.set_title("Dark cluster mask (k-means)")
            ax.axis("off")
            self.show_plot_main(fig)

        _, brain_bin = cv2.threshold(
            img_denoised, 0, 255,
            cv2.THRESH_BINARY + cv2.THRESH_OTSU
        )
        brain_mask = (brain_bin > 0).astype(np.uint8)

        x, y, w_b, h_b = cv2.boundingRect(brain_bin)

        if debug_plots:
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(brain_mask, cmap='gray')
            ax.set_title("Brain mask (Otsu)")
            ax.axis("off")
            self.show_plot_main(fig)

        candidate_mask = np.zeros_like(dark_mask)
        candidate_mask[y:y + h_b, x:x + w_b] = dark_mask[y:y + h_b, x:x + w_b]

        kernel = np.ones((3, 3), np.uint8)
        candidate_mask = cv2.morphologyEx(candidate_mask, cv2.MORPH_OPEN, kernel, iterations=1)

        if debug_plots:
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(candidate_mask, cmap='gray')
            ax.set_title("Dark mask restricted to brain")
            ax.axis("off")
            self.show_plot_main(fig)

 
        num_labels, cc_labels, stats, centroids = cv2.connectedComponentsWithStats(
            candidate_mask, connectivity=8
        )

        ventricle_mask = np.zeros_like(candidate_mask, dtype=np.uint8)

        max_area = 0.0
        perimeter = 0.0
        circularity = np.nan
        eccentricity = np.nan
        solidity = np.nan
        extent = np.nan
        mean_intensity = np.nan

        ventricle_label = -1

        for lab in range(1, num_labels):
            area = stats[lab, cv2.CC_STAT_AREA]
            cx, cy = centroids[lab]

            if (x <= cx <= x + w_b) and (y <= cy <= y + h_b):
                if area > max_area:
                    max_area = float(area)
                    ventricle_label = lab

        if ventricle_label > 0:
            ventricle_mask_full = (cc_labels == ventricle_label).astype(np.uint8)

            kernel = np.ones((3, 3), np.uint8)
            dilated = cv2.dilate(ventricle_mask_full, kernel, iterations=1)
            ventricle_contour = (dilated - ventricle_mask_full)
            perimeter = float(np.sum(ventricle_contour))

            circularity = (4.0 * np.pi * max_area) / (perimeter ** 2 + 1e-8)

            labeled = label(ventricle_mask_full)
            regions = regionprops(labeled, intensity_image=img_norm)

            if len(regions) > 0:
                r = regions[0]
                eccentricity  = float(r.eccentricity)
                solidity      = float(r.solidity)
                extent        = float(r.extent)
                mean_intensity = float(r.mean_intensity)

            ventricle_mask[ventricle_contour == 1] = 1

        features = {
            "area": max_area,
            "perimeter": perimeter,
            "circularity": circularity,
            "eccentricity": eccentricity,
            "solidity": solidity,
            "extent": extent,
            "mean_intensity": mean_intensity,
        }

        if debug_plots:
            fig = Figure(figsize=(5, 5), dpi=100)
            ax = fig.add_subplot(111)
            ax.imshow(ventricle_mask, cmap='gray')
            ax.set_title(
                f"Ventricle mask\n" 
                f"area={max_area:.1f}, P={perimeter:.1f}, circ={circularity:.3f}\n" 
                f"ecc={eccentricity:.3f}, sol={solidity:.3f}, ext={extent:.3f}"
            )
            ax.axis("off")
            self.show_plot_main(fig)

            overlay = cv2.cvtColor(img_norm, cv2.COLOR_GRAY2BGR)
            overlay[ventricle_mask == 1] = [255, 0, 0]

            fig2 = Figure(figsize=(5, 5), dpi=100)
            ax2 = fig2.add_subplot(111)
            ax2.imshow(overlay)
            ax2.set_title("Ventricle classification (red)")
            ax2.axis("off")
            self.show_plot_main(fig2)

        return ventricle_mask, label_img, centers, features

    def segmentar_ventriculos(self):
        if self.current_image_data is None:
            messagebox.showwarning("Aviso", "Carregue uma imagem primeiro!")
            return
        try:
            slice_data = self.current_image_data
            
            ventricle_mask, label_img, centers, features = self.identify_ventricles(
                slice_data, idx=0, debug_plots=True
            )
            
            info_text = (
                f"Características dos Ventrículos:\n\n"
                f"Área: {features['area']:.1f} pixels\n"
                f"Perímetro: {features['perimeter']:.1f}\n"
                f"Circularidade: {features['circularity']:.3f}\n"
                f"Excentricidade: {features['eccentricity']:.3f}\n"
                f"Solidez: {features['solidity']:.3f}\n"
                f"Extensão: {features['extent']:.3f}\n"
                f"Intensidade Média: {features['mean_intensity']:.1f}"
            )
            messagebox.showinfo("Análise de Ventrículos", info_text)
            
        except Exception as e:
            messagebox.showerror("Erro", f"Falha na segmentação:\n{str(e)}")

    def atualizar_fontes(self):
        self.base_font = ("Arial", self.font_size)
        self.style.configure("TButton", font=self.base_font)
        self.style.configure("TLabel", font=self.base_font)
        
        for widget in [self.lbl_arquivo, self.lbl_formato, self.lbl_dimensoes, 
                      self.lbl_zoom]:
            widget.config(font=self.base_font)
    
    def aumentar_fonte(self):
        self.font_size += 2
        self.atualizar_fontes()
        messagebox.showinfo("Acessibilidade", f"Fonte aumentada para {self.font_size}pt")
    
    def diminuir_fonte(self):
        if self.font_size > 8:
            self.font_size -= 2
            self.atualizar_fontes()
            messagebox.showinfo("Acessibilidade", f"Fonte reduzida para {self.font_size}pt")

if __name__ == "__main__":
    root = tk.Tk()
    app = AlzheimerAnalysisGUI(root)
    root.mainloop()
