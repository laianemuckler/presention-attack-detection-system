"""
Módulo de Classificação para Detecção de Ataques
Implementa Random Forest e SVM para classificação Real vs Fake
"""

import numpy as np
import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, classification_report, roc_auc_score, roc_curve
)
from typing import Tuple, Dict, Optional
import logging
from pathlib import Path

from face_detector import FaceDetector
from feature_extraction import FeatureExtractor

logger = logging.getLogger(__name__)


class FaceAntiSpoofing:
    """
    Sistema completo de detecção de ataques de apresentação facial.
    
    Pipeline:
    1. Detectar face na imagem
    2. Extrair features (bordas, textura, sharpness)
    3. Classificar como Real ou Fake
    """
    
    def __init__(self, 
                 model_path: Optional[str] = None,
                 classifier_type: str = 'random_forest',
                 use_fft: bool = False):
        """
        Inicializa o sistema de anti-spoofing.
        
        Args:
            model_path: Caminho para modelo treinado (None = criar novo)
            classifier_type: 'random_forest' ou 'svm'
            use_fft: Se True, usa features de frequência
        """
        self.classifier_type = classifier_type
        self.use_fft = use_fft
        
        # Inicializar componentes
        self.face_detector = FaceDetector()
        self.feature_extractor = FeatureExtractor(use_fft=use_fft)
        self.scaler = StandardScaler()
        
        # Carregar ou criar classificador
        if model_path and Path(model_path).exists():
            self.load_model(model_path)
        else:
            self.classifier = self._create_classifier()
            self.is_trained = False
    
    def _create_classifier(self):
        """
        Cria classificador baseado no tipo especificado.
        
        Returns:
            Classificador não treinado
        """
        if self.classifier_type == 'random_forest':
            return RandomForestClassifier(
                n_estimators=100,
                max_depth=20,
                min_samples_split=5,
                min_samples_leaf=2,
                random_state=42,
                n_jobs=-1,
                class_weight='balanced'  # Importante para datasets desbalanceados
            )
        elif self.classifier_type == 'svm':
            return SVC(
                kernel='rbf',
                C=1.0,
                gamma='scale',
                probability=True,  # Necessário para scores de confiança
                random_state=42,
                class_weight='balanced'
            )
        else:
            raise ValueError(f"Tipo de classificador inválido: {self.classifier_type}")
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, 
              X_val: Optional[np.ndarray] = None, 
              y_val: Optional[np.ndarray] = None,
              optimize_hyperparams: bool = False) -> Dict:
        """
        Treina o classificador.
        
        Args:
            X_train: Features de treinamento (n_samples, n_features)
            y_train: Labels (0=fake, 1=real)
            X_val: Features de validação (opcional)
            y_val: Labels de validação (opcional)
            optimize_hyperparams: Se True, otimiza hiperparâmetros com GridSearch
            
        Returns:
            Dicionário com métricas de treinamento
        """
        logger.info(f"Iniciando treinamento com {len(X_train)} amostras...")
        
        # Normalizar features
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        # Otimizar hiperparâmetros se solicitado
        if optimize_hyperparams:
            logger.info("Otimizando hiperparâmetros...")
            self.classifier = self._optimize_hyperparameters(X_train_scaled, y_train)
        
        # Treinar modelo
        self.classifier.fit(X_train_scaled, y_train)
        self.is_trained = True
        
        # Avaliar no conjunto de treinamento
        train_pred = self.classifier.predict(X_train_scaled)
        train_metrics = {
            'train_accuracy': accuracy_score(y_train, train_pred),
            'train_precision': precision_score(y_train, train_pred),
            'train_recall': recall_score(y_train, train_pred),
            'train_f1': f1_score(y_train, train_pred)
        }
        
        logger.info(f"Train Accuracy: {train_metrics['train_accuracy']:.4f}")
        
        # Avaliar no conjunto de validação se fornecido
        if X_val is not None and y_val is not None:
            X_val_scaled = self.scaler.transform(X_val)
            val_pred = self.classifier.predict(X_val_scaled)
            val_metrics = {
                'val_accuracy': accuracy_score(y_val, val_pred),
                'val_precision': precision_score(y_val, val_pred),
                'val_recall': recall_score(y_val, val_pred),
                'val_f1': f1_score(y_val, val_pred)
            }
            train_metrics.update(val_metrics)
            logger.info(f"Val Accuracy: {val_metrics['val_accuracy']:.4f}")
        
        # Cross-validation score
        cv_scores = cross_val_score(self.classifier, X_train_scaled, y_train, cv=5)
        train_metrics['cv_mean'] = cv_scores.mean()
        train_metrics['cv_std'] = cv_scores.std()
        
        logger.info(f"CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        return train_metrics
    
    def _optimize_hyperparameters(self, X: np.ndarray, y: np.ndarray):
        """
        Otimiza hiperparâmetros usando GridSearchCV.
        
        Args:
            X: Features
            y: Labels
            
        Returns:
            Melhor classificador encontrado
        """
        if self.classifier_type == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [10, 20, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
        else:  # SVM
            param_grid = {
                'C': [0.1, 1, 10],
                'gamma': ['scale', 'auto', 0.001, 0.01],
                'kernel': ['rbf', 'poly']
            }
        
        grid_search = GridSearchCV(
            self._create_classifier(),
            param_grid,
            cv=5,
            scoring='f1',
            n_jobs=-1,
            verbose=1
        )
        
        grid_search.fit(X, y)
        
        logger.info(f"Melhores parâmetros: {grid_search.best_params_}")
        logger.info(f"Melhor score: {grid_search.best_score_:.4f}")
        
        return grid_search.best_estimator_
    
    def predict(self, image_path_or_array) -> Tuple[str, float, Dict]:
        """
        Prediz se uma imagem é real ou fake.
        
        Args:
            image_path_or_array: Caminho da imagem ou array numpy
            
        Returns:
            Tupla (prediction, confidence, details)
            - prediction: 'real' ou 'fake'
            - confidence: Score de confiança [0, 1]
            - details: Dict com informações adicionais
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não foi treinado ainda!")
        
        # Carregar imagem se necessário
        if isinstance(image_path_or_array, str):
            import cv2
            image = cv2.imread(image_path_or_array)
            if image is None:
                raise ValueError(f"Não foi possível carregar imagem: {image_path_or_array}")
        else:
            image = image_path_or_array
        
        # Detectar face
        face_result = self.face_detector.detect_face(image)
        
        if face_result is None:
            return 'unknown', 0.0, {'error': 'Nenhuma face detectada'}
        
        face_roi, face_info = face_result
        
        # Validar qualidade da face
        is_valid, reason = self.face_detector.validate_face_quality(face_roi)
        if not is_valid:
            logger.warning(f"Qualidade da face baixa: {reason}")
        
        # Preprocessar face
        face_processed = self.face_detector.preprocess_face(face_roi)
        
        # Extrair features
        features = self.feature_extractor.extract_all_features(face_processed)
        features_scaled = self.scaler.transform(features.reshape(1, -1))
        
        # Predição com threshold customizado
        prediction_proba = self.classifier.predict_proba(features_scaled)[0]
        threshold = 0.7  # Threshold padrão
        prediction_label = 1 if prediction_proba[1] >= threshold else 0
        
        prediction = 'real' if prediction_label == 1 else 'fake'
        confidence = prediction_proba[prediction_label]
        
        # Detalhes adicionais
        details = {
            'face_bbox': face_info['bbox'],
            'face_confidence': face_info['confidence'],
            'face_quality': reason,
            'probability_real': prediction_proba[1],
            'probability_fake': prediction_proba[0],
            'features_extracted': len(features)
        }
        
        return prediction, confidence, details
    
    def evaluate(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict:
        """
        Avalia o modelo em conjunto de teste.
        
        Args:
            X_test: Features de teste
            y_test: Labels verdadeiros
            
        Returns:
            Dicionário com todas as métricas
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não foi treinado ainda!")
        
        # Normalizar
        X_test_scaled = self.scaler.transform(X_test)
        
        # Predições com threshold customizado
        y_proba = self.classifier.predict_proba(X_test_scaled)[:, 1]
        threshold = 0.7  # Mesmo threshold usado no predict()
        y_pred = (y_proba >= threshold).astype(int)
        
        # Métricas
        metrics = {
            'accuracy': accuracy_score(y_test, y_pred),
            'precision': precision_score(y_test, y_pred),
            'recall': recall_score(y_test, y_pred),
            'f1_score': f1_score(y_test, y_pred),
            'roc_auc': roc_auc_score(y_test, y_proba),
            'confusion_matrix': confusion_matrix(y_test, y_pred).tolist(),
            'classification_report': classification_report(y_test, y_pred, 
                                                          target_names=['fake', 'real'])
        }
        
        # FAR e FRR (métricas importantes para biometria)
        tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()
        metrics['FAR'] = fp / (fp + tn)  # False Acceptance Rate
        metrics['FRR'] = fn / (fn + tp)  # False Rejection Rate
        
        logger.info(f"Test Accuracy: {metrics['accuracy']:.4f}")
        logger.info(f"FAR: {metrics['FAR']:.4f}, FRR: {metrics['FRR']:.4f}")
        
        return metrics
    
    def get_feature_importance(self) -> Optional[np.ndarray]:
        """
        Retorna importância das features (apenas para Random Forest).
        
        Returns:
            Array com importâncias ou None se não aplicável
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não foi treinado ainda!")
        
        if hasattr(self.classifier, 'feature_importances_'):
            return self.classifier.feature_importances_
        else:
            logger.warning("Classificador não suporta feature importance")
            return None
    
    def save_model(self, save_path: str):
        """
        Salva modelo treinado em disco.
        
        Args:
            save_path: Caminho para salvar o modelo
        """
        if not self.is_trained:
            raise RuntimeError("Modelo não foi treinado ainda!")
        
        model_data = {
            'classifier': self.classifier,
            'scaler': self.scaler,
            'classifier_type': self.classifier_type,
            'use_fft': self.use_fft,
            'feature_extractor_params': {
                'lbp_radius': self.feature_extractor.lbp_radius,
                'lbp_points': self.feature_extractor.lbp_points
            }
        }
        
        joblib.dump(model_data, save_path)
        logger.info(f"Modelo salvo em: {save_path}")
    
    def load_model(self, model_path: str):
        """
        Carrega modelo treinado do disco.
        
        Args:
            model_path: Caminho do modelo salvo
        """
        model_data = joblib.load(model_path)
        
        self.classifier = model_data['classifier']
        self.scaler = model_data['scaler']
        self.classifier_type = model_data['classifier_type']
        self.use_fft = model_data['use_fft']
        
        # Recriar feature extractor com parâmetros corretos
        fe_params = model_data['feature_extractor_params']
        self.feature_extractor = FeatureExtractor(
            lbp_radius=fe_params['lbp_radius'],
            lbp_points=fe_params['lbp_points'],
            use_fft=self.use_fft
        )
        
        self.is_trained = True
        logger.info(f"Modelo carregado de: {model_path}")


def test_classifier():
    """
    Função de teste para o classificador.
    """
    from sklearn.datasets import make_classification
    
    # Criar dataset sintético
    X, y = make_classification(
        n_samples=1000,
        n_features=30,
        n_informative=20,
        n_redundant=5,
        random_state=42
    )
    
    # Split train/test
    split_idx = int(0.8 * len(X))
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]
    
    # Testar com Random Forest
    print("=== Testando Random Forest ===")
    clf_rf = FaceAntiSpoofing(classifier_type='random_forest')
    clf_rf.classifier = clf_rf._create_classifier()
    clf_rf.is_trained = False
    
    # Simular treinamento direto (sem pipeline completo)
    train_metrics = clf_rf.train(X_train, y_train, X_test, y_test)
    print(f"Métricas de treino: {train_metrics}")
    
    # Testar com SVM
    print("\n=== Testando SVM ===")
    clf_svm = FaceAntiSpoofing(classifier_type='svm')
    clf_svm.classifier = clf_svm._create_classifier()
    clf_svm.is_trained = False
    
    train_metrics = clf_svm.train(X_train, y_train, X_test, y_test)
    print(f"Métricas de treino: {train_metrics}")
    
    print("\nTeste concluído!")


if __name__ == "__main__":
    test_classifier()
