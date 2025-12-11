"""
Funções auxiliares para o sistema de detecção de ataques
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Tuple, Optional
import logging

# Configurar logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_image(image_path: str) -> Optional[np.ndarray]:
    """
    Carrega imagem do disco.
    
    Args:
        image_path: Caminho para a imagem
        
    Returns:
        Imagem em formato numpy array (BGR) ou None se falhar
    """
    try:
        image = cv2.imread(image_path)
        if image is None:
            logger.error(f"Não foi possível carregar a imagem: {image_path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Erro ao carregar imagem: {e}")
        return None


def resize_image(image: np.ndarray, target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
    """
    Redimensiona imagem mantendo aspect ratio.
    
    Args:
        image: Imagem original
        target_size: Tamanho desejado (width, height)
        
    Returns:
        Imagem redimensionada
    """
    return cv2.resize(image, target_size, interpolation=cv2.INTER_AREA)


def normalize_image(image: np.ndarray) -> np.ndarray:
    """
    Normaliza valores de pixel para [0, 1].
    
    Args:
        image: Imagem original
        
    Returns:
        Imagem normalizada
    """
    return image.astype(np.float32) / 255.0


def convert_to_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converte imagem para escala de cinza.
    
    Args:
        image: Imagem BGR
        
    Returns:
        Imagem em grayscale
    """
    if len(image.shape) == 2:
        return image
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)


def visualize_edges(image: np.ndarray, edges: np.ndarray) -> np.ndarray:
    """
    Cria visualização lado a lado: original + bordas.
    
    Args:
        image: Imagem original
        edges: Bordas detectadas
        
    Returns:
        Imagem concatenada
    """
    # Converter bordas para BGR para visualização
    edges_bgr = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Redimensionar se necessário
    if image.shape[:2] != edges.shape[:2]:
        edges_bgr = cv2.resize(edges_bgr, (image.shape[1], image.shape[0]))
    
    return np.hstack([image, edges_bgr])


def calculate_sharpness(image: np.ndarray) -> float:
    """
    Calcula nitidez da imagem usando variância do Laplaciano.
    
    Args:
        image: Imagem (pode ser colorida ou grayscale)
        
    Returns:
        Score de nitidez (valores maiores = mais nítido)
    """
    gray = convert_to_grayscale(image) if len(image.shape) == 3 else image
    laplacian = cv2.Laplacian(gray, cv2.CV_64F)
    return laplacian.var()


def create_dataset_structure(base_path: str):
    """
    Cria estrutura de diretórios para o dataset.
    
    Args:
        base_path: Caminho base do projeto
    """
    paths = [
        f"{base_path}/data/raw",
        f"{base_path}/data/processed",
        f"{base_path}/data/train/real",
        f"{base_path}/data/train/fake",
        f"{base_path}/data/test/real",
        f"{base_path}/data/test/fake",
        f"{base_path}/models",
        f"{base_path}/notebooks",
        f"{base_path}/docs",
        f"{base_path}/scripts",
        f"{base_path}/tests"
    ]
    
    for path in paths:
        Path(path).mkdir(parents=True, exist_ok=True)
        logger.info(f"Diretório criado: {path}")


def apply_augmentation(image: np.ndarray, augmentation_type: str = 'flip') -> np.ndarray:
    """
    Aplica data augmentation na imagem.
    
    Args:
        image: Imagem original
        augmentation_type: Tipo de augmentation ('flip', 'rotate', 'brightness')
        
    Returns:
        Imagem augmentada
    """
    if augmentation_type == 'flip':
        return cv2.flip(image, 1)
    
    elif augmentation_type == 'rotate':
        angle = np.random.randint(-15, 15)
        h, w = image.shape[:2]
        M = cv2.getRotationMatrix2D((w/2, h/2), angle, 1.0)
        return cv2.warpAffine(image, M, (w, h))
    
    elif augmentation_type == 'brightness':
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        factor = np.random.uniform(0.7, 1.3)
        hsv[:,:,2] = np.clip(hsv[:,:,2] * factor, 0, 255)
        return cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    return image


def get_face_region_of_interest(image: np.ndarray, bbox: Tuple[int, int, int, int]) -> np.ndarray:
    """
    Extrai região de interesse (ROI) da face.
    
    Args:
        image: Imagem completa
        bbox: Bounding box (x, y, width, height)
        
    Returns:
        ROI da face
    """
    x, y, w, h = bbox
    return image[y:y+h, x:x+w]


def display_prediction_result(image: np.ndarray, prediction: str, confidence: float) -> np.ndarray:
    """
    Adiciona texto de predição na imagem.
    
    Args:
        image: Imagem original
        prediction: "real" ou "fake"
        confidence: Score de confiança (0-1)
        
    Returns:
        Imagem com texto sobreposto
    """
    result_img = image.copy()
    
    # Definir cor baseado na predição
    color = (0, 255, 0) if prediction == "real" else (0, 0, 255)
    label = f"{prediction.upper()} ({confidence:.2%})"
    
    # Adicionar retângulo de fundo
    (w, h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.rectangle(result_img, (10, 10), (20 + w, 30 + h), color, -1)
    
    # Adicionar texto
    cv2.putText(result_img, label, (15, 35), 
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    return result_img
