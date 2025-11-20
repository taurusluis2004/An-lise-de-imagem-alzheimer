import tkinter as tk
from tkinter import ttk, messagebox, filedialog
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure
import numpy as np
from image_loader import ImageLoader
import os

class AppGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Sistema de Análise de Imagens - Alzheimer")
        self.root.geometry("1200x800")
        
        # Variáveis de controle
        self.current_image_data = None
        self.current_metadata = None
        self.current_slice_index = 0
        self.current_axis = 2  # 0=sagital, 1=coronal, 2=axial
        self.zoom_level = 1.0
        
        # Dataset paths
        self.dataset_path = os.path.join(os.path.dirname(__file__), 'axl')
        
        # Preparação para modelos do grupo (DS=2, NR=1, NC=1, ND=1)
        self.svm_model = None           # NC=1: SVM (classificador raso)
        self.xgboost_model = None       # NR=1: XGBoost (regressor raso)
        self.densenet_classif = None    # ND=1: DenseNet (classificador profundo)
        self.densenet_regress = None    # ND=1: DenseNet (regressor profundo)
        self.data_processor = None

        # -----------------------------
        # FONTES (ACESSIBILIDADE)
        # -----------------------------
        self.font_size = 12
        self.base_font = ("Arial", self.font_size)

        # Estilo global
        self.style = ttk.Style()
        self.style.configure("TButton", font=self.base_font, padding=5)
        self.style.configure("TLabel", font=self.base_font)
        self.style.configure("TMenu", font=self.base_font)

        # -----------------------------
        # MENU SUPERIOR
        # -----------------------------
        menu_bar = tk.Menu(self.root)

        # Menu Arquivo
        menu_arquivo = tk.Menu(menu_bar, tearoff=0)
        menu_arquivo.add_command(label="Carregar imagem axial", command=self.carregar_dataset)
        menu_arquivo.add_separator()
        menu_arquivo.add_command(label="Sair", command=self.root.quit)
        menu_bar.add_cascade(label="Arquivo", menu=menu_arquivo)
        
        # Menu Dados
        menu_dados = tk.Menu(menu_bar, tearoff=0)
        menu_dados.add_command(label="Preparar dados (80/20)", command=self.preparar_dados)
        menu_dados.add_command(label="Extrair características", command=self.extrair_caracteristicas)
        menu_bar.add_cascade(label="Dados", menu=menu_dados)

        # Menu SVM (NC=1)
        menu_svm = tk.Menu(menu_bar, tearoff=0)
        menu_svm.add_command(label="Treinar SVM", command=self.treinar_svm)
        menu_svm.add_command(label="Avaliar SVM", command=self.avaliar_svm)
        menu_svm.add_command(label="Matriz de Confusão", command=self.matriz_confusao_svm)
        menu_bar.add_cascade(label="SVM (NC=1)", menu=menu_svm)
        
        # Menu XGBoost (NR=1)
        menu_xgb = tk.Menu(menu_bar, tearoff=0)
        menu_xgb.add_command(label="Treinar XGBoost", command=self.treinar_xgboost)
        menu_xgb.add_command(label="Avaliar XGBoost", command=self.avaliar_xgboost)
        menu_xgb.add_command(label="Análise Temporal", command=self.analise_temporal_xgb)
        menu_bar.add_cascade(label="XGBoost (NR=1)", menu=menu_xgb)
        
        # Menu DenseNet (ND=1)
        menu_densenet = tk.Menu(menu_bar, tearoff=0)
        menu_densenet.add_command(label="Treinar Classificação", command=self.treinar_densenet_classif)
        menu_densenet.add_command(label="Treinar Regressão", command=self.treinar_densenet_regress)
        menu_densenet.add_separator()
        menu_densenet.add_command(label="Avaliar Classificação", command=self.avaliar_densenet_classif)
        menu_densenet.add_command(label="Avaliar Regressão", command=self.avaliar_densenet_regress)
        menu_densenet.add_separator()
        menu_densenet.add_command(label="Matriz de Confusão", command=self.matriz_confusao_densenet)
        menu_densenet.add_command(label="Análise Temporal", command=self.analise_temporal_densenet)
        menu_bar.add_cascade(label="DenseNet (ND=1)", menu=menu_densenet)

        # Menu Acessibilidade
        menu_acess = tk.Menu(menu_bar, tearoff=0)
        menu_acess.add_command(label="Aumentar fonte", command=self.aumentar_fonte)
        menu_acess.add_command(label="Diminuir fonte", command=self.diminuir_fonte)
        menu_bar.add_cascade(label="Acessibilidade", menu=menu_acess)

        self.root.config(menu=menu_bar)

        # -----------------------------
        # LAYOUT PRINCIPAL - Dividido em 2 painéis
        # -----------------------------
        # Painel esquerdo - Controles e informações
        self.painel_esquerdo = ttk.Frame(self.root, width=300)
        self.painel_esquerdo.pack(side=tk.LEFT, fill=tk.BOTH, padx=10, pady=10)
        
        # Painel direito - Visualização da imagem
        self.painel_direito = ttk.Frame(self.root)
        self.painel_direito.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10, pady=10)

        # -----------------------------
        # PAINEL ESQUERDO - Controles
        # -----------------------------
        # Informações da imagem
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
        
        # Controles de navegação (para imagens 3D - APENAS AXIAL)
        self.nav_frame = ttk.LabelFrame(self.painel_esquerdo, text="Navegação - Plano Axial (DS=2)", padding=10)
        self.nav_frame.pack(fill=tk.X, pady=10)
        
        ttk.Label(self.nav_frame, text="Fatia Axial:").pack(anchor=tk.W, pady=(5, 0))
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

        # -----------------------------
        # PAINEL DIREITO - Visualização
        # -----------------------------
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.ax.set_title("Visualizador de Imagens Médicas")
        self.ax.axis('off')
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.painel_direito)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)
        
        # Mensagem inicial
        self.ax.text(0.5, 0.5, 'Carregue uma imagem\npelo menu Arquivo', 
                    ha='center', va='center', fontsize=16, transform=self.ax.transAxes)

    # -----------------------------
    # Funções de carregamento de imagem
    # -----------------------------
    def carregar_dataset(self):
        """Carrega uma imagem do dataset AXIAL (DS=2)"""
        if not os.path.exists(self.dataset_path):
            messagebox.showerror("Erro", f"Diretório axial não encontrado: {self.dataset_path}")
            return
        
        files = [f for f in os.listdir(self.dataset_path) if f.endswith('.nii')]
        if not files:
            messagebox.showerror("Erro", "Nenhuma imagem .nii encontrada no diretório axl/")
            return
        
        # Janela de seleção
        dialog = tk.Toplevel(self.root)
        dialog.title("Dataset Axial (DS=2)")
        dialog.geometry("400x500")
        
        ttk.Label(dialog, text="Imagens Axiais (.nii):", font=("Arial", 12, "bold")).pack(pady=10)
        
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
    
    def carregar_e_exibir(self, filepath):
        """Carrega e exibe uma imagem"""
        try:
            self.current_image_data, self.current_metadata = ImageLoader.load_image(filepath)
            
            # Atualiza informações
            self.lbl_arquivo.config(text=f"Arquivo: {os.path.basename(filepath)}")
            self.lbl_formato.config(text=f"Formato: {self.current_metadata['format']}")
            self.lbl_dimensoes.config(text=f"Dimensões: {self.current_metadata['shape']}")
            
            # Configura slider para imagens 3D
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
        
        # Obtém fatia para exibir
        if len(self.current_image_data.shape) == 3:
            slice_data = ImageLoader.get_slice(self.current_image_data, 
                                              self.current_axis, 
                                              self.current_slice_index)
            max_slice = self.current_image_data.shape[self.current_axis] - 1
            self.lbl_slice.config(text=f"Fatia: {self.current_slice_index}/{max_slice}")
        else:
            slice_data = self.current_image_data
        
        # Normaliza para exibição
        display_data = ImageLoader.normalize_for_display(slice_data)
        
        # Exibe com zoom
        self.ax.imshow(display_data, cmap='gray')
        self.ax.set_title(f"{os.path.basename(self.current_metadata['file_path'])} - Zoom: {int(self.zoom_level*100)}%")
        self.ax.axis('off')
        
        # Aplica zoom
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
    
    # -----------------------------
    # Controles de navegação e zoom
    # -----------------------------
    def mudar_fatia(self, value):
        """Muda a fatia exibida"""
        self.current_slice_index = int(float(value))
        self.exibir_imagem()
    
    def zoom_in(self):
        """Aumenta o zoom"""
        self.zoom_level *= 1.2
        self.lbl_zoom.config(text=f"Zoom: {int(self.zoom_level*100)}%")
        self.exibir_imagem()
    
    def zoom_out(self):
        """Diminui o zoom"""
        self.zoom_level /= 1.2
        if self.zoom_level < 0.5:
            self.zoom_level = 0.5
        self.lbl_zoom.config(text=f"Zoom: {int(self.zoom_level*100)}%")
        self.exibir_imagem()
    
    def zoom_reset(self):
        """Reseta o zoom"""
        self.zoom_level = 1.0
        self.lbl_zoom.config(text=f"Zoom: {int(self.zoom_level*100)}%")
        self.exibir_imagem()
    
    # -----------------------------
    # Funções de pré-processamento (PREPARADAS PARA IMPLEMENTAÇÃO)
    # -----------------------------
    def preparar_dados(self):
        """
        Prepara dados com split 80/20 por paciente (DS=2 - Axial).
        
        REQUISITOS:
        - Dataset Axial (DS=2) apenas
        - Divide PACIENTES (não exames) em treino (80%) e teste (20%)
        - 2 classes: Demented e NonDemented
        - Converted: CDR=0 → NonDemented, CDR>0 → Demented
        - Balanceamento 4:1 entre treino e teste
        - Validação: 20% do treino
        - SEM MISTURA de exames do mesmo paciente entre treino/teste
        """
        messagebox.showinfo("Preparar Dados (DS=2 - Axial)", 
                          "FUNÇÃO A IMPLEMENTAR:\n\n"
                          "1. Carregar oasis_longitudinal_demographic.csv\n"
                          "2. Mapear Group:\n"
                          "   - Nondemented → NonDemented\n"
                          "   - Demented → Demented\n"
                          "   - Converted: se CDR=0 → NonDemented, se CDR>0 → Demented\n"
                          "3. Listar Subject IDs únicos\n"
                          "4. Split 80/20 por Subject ID (não por exame!)\n"
                          "5. Balancear classes 4:1 (treino:teste)\n"
                          "6. Validação: 20% do conjunto de treino\n"
                          "7. Salvar splits para uso nos modelos")
    
    def extrair_caracteristicas(self):
        """
        Extrai características das imagens axiais para modelos rasos (SVM e XGBoost).
        """
        messagebox.showinfo("Extrair Características", 
                          "FUNÇÃO A IMPLEMENTAR:\n\n"
                          "Do CSV:\n"
                          "- nWBV (volume cerebral normalizado)\n"
                          "- eTIV (volume intracraniano total)\n"
                          "- MMSE, Age, EDUC, SES\n\n"
                          "Das imagens (.nii):\n"
                          "- Estatísticas (média, desvio, min, max)\n"
                          "- Texturas (Haralick, GLCM)\n"
                          "- Histogramas")
    
    # -----------------------------
    # SVM (NC=1) - Classificador Raso
    # -----------------------------
    def treinar_svm(self):
        """Treina SVM (NC=1) com características extraídas"""
        messagebox.showinfo("SVM (NC=1)", 
                          "Treinar classificador SVM:\n\n"
                          "• Entrada: Características extraídas\n"
                          "• Saída: Demented / NonDemented\n"
                          "• Hiperparâmetros: Grid Search (C, gamma, kernel)\n"
                          "• Validação: 20% do treino")
    
    def avaliar_svm(self):
        """Avalia SVM no conjunto de teste"""
        messagebox.showinfo("Avaliar SVM", 
                          "Métricas no TESTE:\n\n"
                          "• Acurácia\n"
                          "• Sensibilidade\n"
                          "• Especificidade")
    
    def matriz_confusao_svm(self):
        """Matriz de confusão do SVM"""
        messagebox.showinfo("Matriz de Confusão - SVM", 
                          "Exibir matriz de confusão 2x2\n"
                          "apenas no conjunto de TESTE")
    
    # -----------------------------
    # XGBoost (NR=1) - Regressor Raso
    # -----------------------------
    def treinar_xgboost(self):
        """Treina XGBoost (NR=1) para regressão de idade"""
        messagebox.showinfo("XGBoost (NR=1)", 
                          "Treinar regressor XGBoost:\n\n"
                          "• Entrada: Características extraídas\n"
                          "• Saída: Idade (anos)\n"
                          "• Hiperparâmetros: Grid Search\n"
                          "• Validação: 20% do treino")
    
    def avaliar_xgboost(self):
        """Avalia XGBoost no conjunto de teste"""
        messagebox.showinfo("Avaliar XGBoost", 
                          "Métricas no TESTE:\n\n"
                          "• MAE (Mean Absolute Error)\n"
                          "• RMSE (Root Mean Squared Error)\n"
                          "• R² Score")
    
    def analise_temporal_xgb(self):
        """Análise temporal para XGBoost"""
        messagebox.showinfo("Análise Temporal - XGBoost", 
                          "Verificar progressão de idade:\n\n"
                          "• Visitas posteriores → idades maiores?\n"
                          "• Diferença média entre visitas\n"
                          "• % de casos corretos")
    
    # -----------------------------
    # DenseNet (ND=1) - Modelo Profundo
    # -----------------------------
    def treinar_densenet_classif(self):
        """Treina DenseNet (ND=1) para classificação"""
        messagebox.showinfo("DenseNet Classificação (ND=1)", 
                          "Treinar DenseNet para classificação:\n\n"
                          "• Entrada: Imagens axiais (.nii)\n"
                          "• Saída: Demented / NonDemented\n"
                          "• Fine-tuning: Pesos ImageNet\n"
                          "• Gráficos: Acurácia e Loss por época")
    
    def treinar_densenet_regress(self):
        """Treina DenseNet (ND=1) para regressão"""
        messagebox.showinfo("DenseNet Regressão (ND=1)", 
                          "Treinar DenseNet para regressão:\n\n"
                          "• Entrada: Imagens axiais (.nii)\n"
                          "• Saída: Idade (anos)\n"
                          "• Fine-tuning: Pesos ImageNet\n"
                          "• Gráficos: Loss por época")
    
    def avaliar_densenet_classif(self):
        """Avalia DenseNet classificação no teste"""
        messagebox.showinfo("Avaliar DenseNet - Classificação", 
                          "Métricas no TESTE:\n\n"
                          "• Acurácia\n"
                          "• Sensibilidade\n"
                          "• Especificidade")
    
    def avaliar_densenet_regress(self):
        """Avalia DenseNet regressão no teste"""
        messagebox.showinfo("Avaliar DenseNet - Regressão", 
                          "Métricas no TESTE:\n\n"
                          "• MAE\n"
                          "• RMSE\n"
                          "• R² Score")
    
    def matriz_confusao_densenet(self):
        """Matriz de confusão do DenseNet"""
        messagebox.showinfo("Matriz de Confusão - DenseNet", 
                          "Exibir matriz de confusão 2x2\n"
                          "apenas no conjunto de TESTE")
    
    def analise_temporal_densenet(self):
        """Análise temporal para DenseNet"""
        messagebox.showinfo("Análise Temporal - DenseNet", 
                          "Verificar progressão de idade:\n\n"
                          "• Visitas posteriores → idades maiores?\n"
                          "• Diferença média entre visitas\n"
                          "• % de casos corretos")

    # -----------------------------
    # ACESSIBILIDADE
    # -----------------------------
    def atualizar_fontes(self):
        self.base_font = ("Arial", self.font_size)
        self.style.configure("TButton", font=self.base_font)
        self.style.configure("TLabel", font=self.base_font)
        self.style.configure("TMenu", font=self.base_font)

    def aumentar_fonte(self):
        self.font_size += 2
        self.atualizar_fontes()

    def diminuir_fonte(self):
        if self.font_size > 8:
            self.font_size -= 2
            self.atualizar_fontes()

# -----------------------------
# MAIN
# -----------------------------
if __name__ == "__main__":
    root = tk.Tk()
    app = AppGUI(root)
    root.mainloop()