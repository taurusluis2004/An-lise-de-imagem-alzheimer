"""
Módulo para carregar imagens médicas em diversos formatos.
Suporta: Nifti (.nii, .nii.gz), PNG, JPG/JPEG
"""

import numpy as np
import nibabel as nib
from PIL import Image
import os


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
                - image_data: numpy array com os dados da imagem
                - metadata: dicionário com informações sobre a imagem
        """
        if not os.path.exists(file_path):
            raise FileNotFoundError(f"Arquivo não encontrado: {file_path}")
        
        ext = os.path.splitext(file_path.lower())[1]
        
        # Verifica se é arquivo .nii.gz
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
        """
        Carrega imagem Nifti (3D ou 4D).
        
        Returns:
            tuple: (image_data, metadata)
        """
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
        """
        Carrega imagem 2D (PNG, JPG).
        
        Returns:
            tuple: (image_data, metadata)
        """
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
        """
        Extrai uma fatia 2D de uma imagem 3D.
        
        Args:
            image_data: Array numpy com dados da imagem
            axis: Eixo da fatia (0=sagital, 1=coronal, 2=axial)
            slice_idx: Índice da fatia (None = meio da imagem)
            
        Returns:
            numpy array 2D com a fatia
        """
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
        """
        Normaliza dados da imagem para exibição (0-255).
        
        Args:
            image_data: Array numpy com dados da imagem
            
        Returns:
            Array numpy normalizado para 0-255
        """
        data = image_data.copy()
        
        # Remove NaN e infinitos
        data = np.nan_to_num(data, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Normaliza para 0-255
        data_min = data.min()
        data_max = data.max()
        
        if data_max > data_min:
            data = ((data - data_min) / (data_max - data_min) * 255).astype(np.uint8)
        else:
            data = np.zeros_like(data, dtype=np.uint8)
        
        return data
