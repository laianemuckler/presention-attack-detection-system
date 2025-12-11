"""
Módulo de Extração de Features para Detecção de Ataques
Implementa técnicas de análise de bordas, textura (LBP) e frequência
"""

import cv2
import numpy as np
from skimage.feature import local_binary_pattern
from scipy import fftpack
from typing import Dict, Tuple
import logging

logger = logging.getLogger(__name__)


class FeatureExtractor:
    """
    Extrator de features para detecção de fotos impressas.
    
    Features implementadas:
    - Densidade de bordas (Canny, Sobel)
    - Local Binary Patterns (LBP)
    - Análise de sharpness (Laplacian)
    - Análise de frequência (FFT) - opcional
    """
    
    def __init__(self, 
                 lbp_radius: int = 1, 
                 lbp_points: int = 8,
                 use_fft: bool = False):
        """
        Inicializa o extrator de features.
        
        Args:
            lbp_radius: Raio para LBP
            lbp_points: Número de pontos para LBP
            use_fft: Se True, inclui análise de frequência FFT
        """
        self.lbp_radius = lbp_radius
        self.lbp_points = lbp_points
        self.use_fft = use_fft
        
    def extract_all_features(self, image: np.ndarray) -> np.ndarray:
        """
        Extrai todas as features da imagem.
        
        Args:
            image: Imagem BGR ou grayscale
            
        Returns:
            Vetor de features concatenado
        """
        # Converter para grayscale se necessário
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        features = []
        
        # 1. Features de bordas
        edge_features = self.extract_edge_features(gray)
        features.extend(edge_features)
        
        # 2. Features de textura (LBP)
        lbp_features = self.extract_lbp_features(gray)
        features.extend(lbp_features)
        
        # 3. Features de sharpness
        sharpness_features = self.extract_sharpness_features(gray)
        features.extend(sharpness_features)
        
        # 4. Features de frequência (opcional)
        if self.use_fft:
            fft_features = self.extract_frequency_features(gray)
            features.extend(fft_features)
            
        return np.array(features)
    
    def extract_edge_features(self, gray_image: np.ndarray) -> list:
        """
        Extrai features baseadas em detecção de bordas.
        
        Fotos impressas tendem a ter:
        - Bordas mais definidas nas regiões periféricas
        - Densidade de bordas diferente
        - Padrões de impressão detectáveis
        
        Args:
            gray_image: Imagem em grayscale
            
        Returns:
            Lista de features de bordas
        """
        features = []
        
        # 1. Canny Edge Detection
        edges_canny = cv2.Canny(gray_image, 100, 200)
        edge_density_total = np.sum(edges_canny > 0) / edges_canny.size
        features.append(edge_density_total)
        
        # 2. Densidade de bordas em regiões periféricas (mais importante!)
        h, w = gray_image.shape
        border_width = int(min(h, w) * 0.1)  # 10% das bordas
        
        # Bordas superiores e inferiores
        top_border = edges_canny[:border_width, :]
        bottom_border = edges_canny[-border_width:, :]
        left_border = edges_canny[:, :border_width]
        right_border = edges_canny[:, -border_width:]
        
        features.append(np.sum(top_border > 0) / top_border.size)
        features.append(np.sum(bottom_border > 0) / bottom_border.size)
        features.append(np.sum(left_border > 0) / left_border.size)
        features.append(np.sum(right_border > 0) / right_border.size)
        
        # 3. Sobel gradients (magnitude)
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        sobel_magnitude = np.sqrt(sobelx**2 + sobely**2)
        
        features.append(np.mean(sobel_magnitude))
        features.append(np.std(sobel_magnitude))
        features.append(np.max(sobel_magnitude))
        
        return features
    
    def extract_lbp_features(self, gray_image: np.ndarray) -> list:
        """
        Extrai features de textura usando Local Binary Patterns (LBP).
        
        LBP captura micropadrões de textura que são diferentes entre:
        - Pele real (textura orgânica)
        - Papel impresso (padrões de impressão, dot patterns)
        
        Args:
            gray_image: Imagem em grayscale
            
        Returns:
            Histograma LBP normalizado
        """
        # Calcular LBP
        lbp = local_binary_pattern(
            gray_image, 
            self.lbp_points, 
            self.lbp_radius, 
            method='uniform'
        )
        
        # Histograma LBP
        n_bins = self.lbp_points + 2  # uniform patterns + non-uniform
        hist, _ = np.histogram(
            lbp.ravel(), 
            bins=n_bins, 
            range=(0, n_bins),
            density=True  # Normalizar
        )
        
        return hist.tolist()
    
    def extract_sharpness_features(self, gray_image: np.ndarray) -> list:
        """
        Extrai features de nitidez/sharpness.
        
        Fotos impressas geralmente têm:
        - Sharpness artificial em algumas regiões
        - Variância diferente do Laplaciano
        
        Args:
            gray_image: Imagem em grayscale
            
        Returns:
            Lista de features de sharpness
        """
        features = []
        
        # 1. Variância do Laplaciano (medida clássica de sharpness)
        laplacian = cv2.Laplacian(gray_image, cv2.CV_64F)
        laplacian_var = laplacian.var()
        features.append(laplacian_var)
        
        # 2. Estatísticas do Laplaciano
        features.append(np.mean(np.abs(laplacian)))
        features.append(np.std(laplacian))
        features.append(np.max(np.abs(laplacian)))
        
        # 3. Tenengrad (outra medida de sharpness)
        sobelx = cv2.Sobel(gray_image, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray_image, cv2.CV_64F, 0, 1, ksize=3)
        tenengrad = np.mean(sobelx**2 + sobely**2)
        features.append(tenengrad)
        
        return features
    
    def extract_frequency_features(self, gray_image: np.ndarray) -> list:
        """
        Extrai features no domínio da frequência usando FFT.
        
        Fotos impressas têm padrões de frequência diferentes devido a:
        - Processo de impressão (dots, grids)
        - Re-captura (moiré patterns)
        
        Args:
            gray_image: Imagem em grayscale
            
        Returns:
            Lista de features de frequência
        """
        features = []
        
        # Aplicar FFT 2D
        fft = fftpack.fft2(gray_image)
        fft_shift = fftpack.fftshift(fft)
        magnitude_spectrum = np.abs(fft_shift)
        
        # Log scale para melhor visualização
        magnitude_spectrum = np.log(magnitude_spectrum + 1)
        
        # Dividir espectro em regiões (baixa, média, alta frequência)
        h, w = magnitude_spectrum.shape
        center_h, center_w = h // 2, w // 2
        
        # Raios para diferentes frequências
        radius_low = int(min(h, w) * 0.1)
        radius_high = int(min(h, w) * 0.4)
        
        # Criar máscaras circulares
        y, x = np.ogrid[:h, :w]
        dist_from_center = np.sqrt((x - center_w)**2 + (y - center_h)**2)
        
        # Baixas frequências (centro)
        low_freq_mask = dist_from_center <= radius_low
        low_freq_energy = np.mean(magnitude_spectrum[low_freq_mask])
        features.append(low_freq_energy)
        
        # Altas frequências (bordas)
        high_freq_mask = dist_from_center >= radius_high
        high_freq_energy = np.mean(magnitude_spectrum[high_freq_mask])
        features.append(high_freq_energy)
        
        # Razão alta/baixa frequência
        features.append(high_freq_energy / (low_freq_energy + 1e-6))
        
        # Energia total no espectro
        features.append(np.mean(magnitude_spectrum))
        features.append(np.std(magnitude_spectrum))
        
        return features
    
    def visualize_edges(self, image: np.ndarray) -> Dict[str, np.ndarray]:
        """
        Gera visualizações das bordas detectadas para debug/apresentação.
        
        Args:
            image: Imagem original (BGR ou grayscale)
            
        Returns:
            Dicionário com diferentes visualizações
        """
        if len(image.shape) == 3:
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray = image.copy()
            
        visualizations = {}
        
        # Canny edges
        visualizations['canny'] = cv2.Canny(gray, 100, 200)
        
        # Sobel X e Y
        sobelx = cv2.Sobel(gray, cv2.CV_64F, 1, 0, ksize=3)
        sobely = cv2.Sobel(gray, cv2.CV_64F, 0, 1, ksize=3)
        
        # Normalizar para visualização
        sobelx_vis = cv2.convertScaleAbs(sobelx)
        sobely_vis = cv2.convertScaleAbs(sobely)
        sobel_combined = cv2.addWeighted(sobelx_vis, 0.5, sobely_vis, 0.5, 0)
        
        visualizations['sobel_x'] = sobelx_vis
        visualizations['sobel_y'] = sobely_vis
        visualizations['sobel_combined'] = sobel_combined
        
        # Laplacian
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        visualizations['laplacian'] = cv2.convertScaleAbs(laplacian)
        
        # LBP
        lbp = local_binary_pattern(gray, self.lbp_points, self.lbp_radius, method='uniform')
        visualizations['lbp'] = (lbp / lbp.max() * 255).astype(np.uint8)
        
        return visualizations
    
    def get_feature_names(self) -> list:
        """
        Retorna nomes das features extraídas (útil para análise).
        
        Returns:
            Lista com nomes das features
        """
        names = [
            # Edge features (9)
            'edge_density_total',
            'edge_density_top',
            'edge_density_bottom',
            'edge_density_left',
            'edge_density_right',
            'sobel_mean',
            'sobel_std',
            'sobel_max',
            
            # LBP histogram (10 para radius=1, points=8)
        ]
        
        n_lbp = self.lbp_points + 2
        names.extend([f'lbp_bin_{i}' for i in range(n_lbp)])
        
        # Sharpness features (5)
        names.extend([
            'laplacian_var',
            'laplacian_mean',
            'laplacian_std',
            'laplacian_max',
            'tenengrad'
        ])
        
        # FFT features (5) - se habilitado
        if self.use_fft:
            names.extend([
                'fft_low_freq',
                'fft_high_freq',
                'fft_ratio',
                'fft_mean',
                'fft_std'
            ])
        
        return names


def test_feature_extraction():
    """
    Função de teste para validar extração de features.
    """
    # Criar imagem de teste
    test_image = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
    
    # Inicializar extrator
    extractor = FeatureExtractor(use_fft=True)
    
    # Extrair features
    features = extractor.extract_all_features(test_image)
    feature_names = extractor.get_feature_names()
    
    print(f"Total de features extraídas: {len(features)}")
    print(f"Nomes das features: {len(feature_names)}")
    print(f"\nPrimeiras 10 features:")
    for name, value in zip(feature_names[:10], features[:10]):
        print(f"  {name}: {value:.4f}")
    
    # Visualizações
    visualizations = extractor.visualize_edges(test_image)
    print(f"\nVisualizações geradas: {list(visualizations.keys())}")


if __name__ == "__main__":
    test_feature_extraction()
