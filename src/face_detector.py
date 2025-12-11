"""
Módulo de Detecção Facial
Utiliza MediaPipe para detectar e alinhar faces em imagens
"""

import cv2
import numpy as np
import mediapipe as mp
from typing import Optional, Tuple, List
import logging

logger = logging.getLogger(__name__)


class FaceDetector:
    """
    Detector de faces usando MediaPipe Face Detection.
    
    Responsabilidades:
    - Detectar faces em imagens
    - Extrair região da face (ROI)
    - Alinhar e normalizar faces
    - Validar qualidade da detecção
    """
    
    def __init__(self, 
                 min_detection_confidence: float = 0.5,
                 model_selection: int = 1):
        """
        Inicializa o detector facial.
        
        Args:
            min_detection_confidence: Confiança mínima para detecção (0-1)
            model_selection: 0 = curta distância (<2m), 1 = longa distância (>2m)
        """
        self.min_detection_confidence = min_detection_confidence
        
        # Inicializar MediaPipe Face Detection
        self.mp_face_detection = mp.solutions.face_detection
        self.face_detection = self.mp_face_detection.FaceDetection(
            min_detection_confidence=min_detection_confidence,
            model_selection=model_selection
        )
        
        self.mp_drawing = mp.solutions.drawing_utils
        
    def detect_face(self, image: np.ndarray) -> Optional[Tuple[np.ndarray, dict]]:
        """
        Detecta e extrai a face principal da imagem.
        
        Args:
            image: Imagem BGR (OpenCV format)
            
        Returns:
            Tupla (face_roi, face_info) ou None se não detectar face
            - face_roi: Região da face extraída
            - face_info: Dict com bbox, landmarks, confidence
        """
        # Converter BGR para RGB (MediaPipe usa RGB)
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        # Detectar faces
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            logger.warning("Nenhuma face detectada na imagem")
            return None
        
        # Usar a primeira face detectada (maior confiança)
        detection = results.detections[0]
        
        # Extrair bounding box
        bbox = self._get_bounding_box(detection, image.shape)
        
        if bbox is None:
            logger.warning("Bounding box inválido")
            return None
        
        # Extrair ROI da face
        x, y, w, h = bbox
        face_roi = image[y:y+h, x:x+w]
        
        # Informações da detecção
        face_info = {
            'bbox': bbox,
            'confidence': detection.score[0],
            'landmarks': self._get_landmarks(detection, image.shape)
        }
        
        logger.info(f"Face detectada com confiança: {face_info['confidence']:.2f}")
        
        return face_roi, face_info
    
    def detect_all_faces(self, image: np.ndarray) -> List[Tuple[np.ndarray, dict]]:
        """
        Detecta todas as faces na imagem.
        
        Args:
            image: Imagem BGR
            
        Returns:
            Lista de tuplas (face_roi, face_info)
        """
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        results = self.face_detection.process(image_rgb)
        
        if not results.detections:
            return []
        
        faces = []
        for detection in results.detections:
            bbox = self._get_bounding_box(detection, image.shape)
            if bbox is None:
                continue
                
            x, y, w, h = bbox
            face_roi = image[y:y+h, x:x+w]
            
            face_info = {
                'bbox': bbox,
                'confidence': detection.score[0],
                'landmarks': self._get_landmarks(detection, image.shape)
            }
            
            faces.append((face_roi, face_info))
        
        return faces
    
    def _get_bounding_box(self, detection, image_shape: tuple) -> Optional[Tuple[int, int, int, int]]:
        """
        Converte bounding box relativo para coordenadas absolutas.
        
        Args:
            detection: Objeto de detecção do MediaPipe
            image_shape: Shape da imagem (height, width, channels)
            
        Returns:
            Tupla (x, y, width, height) ou None se inválido
        """
        h, w, _ = image_shape
        bbox = detection.location_data.relative_bounding_box
        
        # Converter coordenadas relativas [0, 1] para pixels
        x = int(bbox.xmin * w)
        y = int(bbox.ymin * h)
        width = int(bbox.width * w)
        height = int(bbox.height * h)
        
        # Validar coordenadas
        if x < 0 or y < 0 or width <= 0 or height <= 0:
            return None
        
        # Garantir que não ultrapasse limites da imagem
        x = max(0, x)
        y = max(0, y)
        width = min(width, w - x)
        height = min(height, h - y)
        
        # Expandir bounding box levemente (10%) para incluir contexto
        padding = int(min(width, height) * 0.1)
        x = max(0, x - padding)
        y = max(0, y - padding)
        width = min(w - x, width + 2 * padding)
        height = min(h - y, height + 2 * padding)
        
        return (x, y, width, height)
    
    def _get_landmarks(self, detection, image_shape: tuple) -> List[Tuple[int, int]]:
        """
        Extrai landmarks (pontos-chave) da face.
        
        Args:
            detection: Objeto de detecção do MediaPipe
            image_shape: Shape da imagem
            
        Returns:
            Lista de tuplas (x, y) com coordenadas dos landmarks
        """
        h, w, _ = image_shape
        landmarks = []
        
        for landmark in detection.location_data.relative_keypoints:
            x = int(landmark.x * w)
            y = int(landmark.y * h)
            landmarks.append((x, y))
        
        return landmarks
    
    def preprocess_face(self, 
                       face_roi: np.ndarray, 
                       target_size: Tuple[int, int] = (224, 224)) -> np.ndarray:
        """
        Preprocessa ROI da face para extração de features.
        
        Args:
            face_roi: Região da face extraída
            target_size: Tamanho desejado (width, height)
            
        Returns:
            Face preprocessada
        """
        # Redimensionar
        face_resized = cv2.resize(face_roi, target_size, interpolation=cv2.INTER_AREA)
        
        # Equalização de histograma (melhora contraste)
        if len(face_resized.shape) == 3:
            # Converter para YUV, equalizar Y, converter de volta
            face_yuv = cv2.cvtColor(face_resized, cv2.COLOR_BGR2YUV)
            face_yuv[:, :, 0] = cv2.equalizeHist(face_yuv[:, :, 0])
            face_resized = cv2.cvtColor(face_yuv, cv2.COLOR_YUV2BGR)
        else:
            face_resized = cv2.equalizeHist(face_resized)
        
        return face_resized
    
    def draw_detections(self, image: np.ndarray, face_info: dict) -> np.ndarray:
        """
        Desenha bounding box e landmarks na imagem (para visualização).
        
        Args:
            image: Imagem original
            face_info: Dicionário com informações da face
            
        Returns:
            Imagem com detecções desenhadas
        """
        annotated_image = image.copy()
        
        # Desenhar bounding box
        x, y, w, h = face_info['bbox']
        cv2.rectangle(annotated_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        
        # Desenhar confiança
        confidence_text = f"Conf: {face_info['confidence']:.2f}"
        cv2.putText(annotated_image, confidence_text, (x, y-10),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Desenhar landmarks
        for landmark in face_info['landmarks']:
            cv2.circle(annotated_image, landmark, 3, (255, 0, 0), -1)
        
        return annotated_image
    
    def validate_face_quality(self, face_roi: np.ndarray) -> Tuple[bool, str]:
        """
        Valida qualidade da face detectada.
        
        Verifica:
        - Tamanho mínimo
        - Blur/nitidez
        - Iluminação
        
        Args:
            face_roi: Região da face
            
        Returns:
            Tupla (is_valid, reason)
        """
        h, w = face_roi.shape[:2]
        
        # 1. Verificar tamanho mínimo
        if h < 80 or w < 80:
            return False, "Face muito pequena (mínimo 80x80)"
        
        # 2. Verificar blur (usando variância do Laplaciano)
        gray = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY) if len(face_roi.shape) == 3 else face_roi
        laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
        
        if laplacian_var < 50:  # Threshold empírico
            return False, f"Imagem muito borrada (var={laplacian_var:.1f})"
        
        # 3. Verificar iluminação
        mean_brightness = np.mean(gray)
        if mean_brightness < 40:
            return False, f"Imagem muito escura (brightness={mean_brightness:.1f})"
        elif mean_brightness > 220:
            return False, f"Imagem muito clara (brightness={mean_brightness:.1f})"
        
        return True, "Face válida"
    
    def close(self):
        """Fecha recursos do detector."""
        self.face_detection.close()
    
    def __del__(self):
        """Destructor para garantir cleanup."""
        try:
            self.close()
        except:
            pass


def test_face_detection():
    """
    Função de teste para validar detecção facial.
    """
    # Criar imagem de teste (face sintética simples)
    test_image = np.ones((480, 640, 3), dtype=np.uint8) * 200
    
    # Simular face (círculo + olhos)
    cv2.circle(test_image, (320, 240), 100, (180, 150, 120), -1)  # Rosto
    cv2.circle(test_image, (290, 220), 15, (0, 0, 0), -1)  # Olho esquerdo
    cv2.circle(test_image, (350, 220), 15, (0, 0, 0), -1)  # Olho direito
    cv2.ellipse(test_image, (320, 270), (40, 20), 0, 0, 180, (100, 50, 50), -1)  # Boca
    
    # Inicializar detector
    detector = FaceDetector()
    
    # Tentar detectar
    result = detector.detect_face(test_image)
    
    if result:
        face_roi, face_info = result
        print(f"Face detectada!")
        print(f"  Confiança: {face_info['confidence']:.2f}")
        print(f"  BBox: {face_info['bbox']}")
        print(f"  Tamanho ROI: {face_roi.shape}")
        
        # Validar qualidade
        is_valid, reason = detector.validate_face_quality(face_roi)
        print(f"  Qualidade: {reason}")
        
        # Preprocessar
        face_processed = detector.preprocess_face(face_roi)
        print(f"  Tamanho preprocessado: {face_processed.shape}")
    else:
        print("Nenhuma face detectada (normal para imagem sintética simples)")
    
    detector.close()
    print("\nTeste concluído!")


if __name__ == "__main__":
    test_face_detection()
