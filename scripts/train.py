"""
Script de Treinamento
Treina classificador Random Forest ou SVM para detecção de ataques
"""

import numpy as np
import argparse
from pathlib import Path
import sys
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import logging
import json

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from classifier import FaceAntiSpoofing
from feature_extraction import FeatureExtractor
from face_detector import FaceDetector
import cv2
from tqdm import tqdm

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def load_images_and_extract_features(data_dir: Path, label: int, 
                                     feature_extractor: FeatureExtractor,
                                     face_detector: FaceDetector):
    """
    Carrega imagens de um diretório e extrai features.
    
    Args:
        data_dir: Diretório com imagens
        label: 0 (fake) ou 1 (real)
        feature_extractor: Instância do FeatureExtractor
        face_detector: Instância do FaceDetector
        
    Returns:
        Arrays X (features) e y (labels)
    """
    logger.info(f"Carregando imagens de {data_dir}")
    
    image_files = list(data_dir.glob("*.jpg")) + list(data_dir.glob("*.png"))
    
    X_list = []
    y_list = []
    
    for img_path in tqdm(image_files, desc=f"Processing {'real' if label==1 else 'fake'}"):
        try:
            # Carregar imagem
            image = cv2.imread(str(img_path))
            if image is None:
                continue
            
            # Se imagem não foi preprocessada, detectar face
            if image.shape[:2] != (224, 224):
                face_result = face_detector.detect_face(image)
                if face_result is None:
                    continue
                face_roi, _ = face_result
                image = face_detector.preprocess_face(face_roi)
            
            # Extrair features
            features = feature_extractor.extract_all_features(image)
            
            X_list.append(features)
            y_list.append(label)
            
        except Exception as e:
            logger.warning(f"Erro processando {img_path}: {e}")
    
    return np.array(X_list), np.array(y_list)


def plot_training_results(metrics: dict, save_dir: Path):
    """
    Plota resultados do treinamento.
    
    Args:
        metrics: Dicionário com métricas
        save_dir: Diretório para salvar gráficos
    """
    save_dir.mkdir(parents=True, exist_ok=True)
    
    # 1. Matriz de confusão
    if 'confusion_matrix' in metrics:
        plt.figure(figsize=(8, 6))
        cm = np.array(metrics['confusion_matrix'])
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=['Fake', 'Real'],
                   yticklabels=['Fake', 'Real'])
        plt.title('Confusion Matrix')
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        plt.tight_layout()
        plt.savefig(save_dir / 'confusion_matrix.png', dpi=300)
        plt.close()
        logger.info(f"Confusion matrix salva em {save_dir / 'confusion_matrix.png'}")
    
    # 2. Métricas comparativas
    if all(k in metrics for k in ['accuracy', 'precision', 'recall', 'f1_score']):
        plt.figure(figsize=(10, 6))
        
        metric_names = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
        metric_values = [
            metrics['accuracy'],
            metrics['precision'],
            metrics['recall'],
            metrics['f1_score']
        ]
        
        colors = ['#2ecc71', '#3498db', '#e74c3c', '#f39c12']
        bars = plt.bar(metric_names, metric_values, color=colors, alpha=0.7)
        
        # Adicionar valores nas barras
        for bar, value in zip(bars, metric_values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylim(0, 1.1)
        plt.ylabel('Score')
        plt.title('Model Performance Metrics')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'metrics_comparison.png', dpi=300)
        plt.close()
        logger.info(f"Métricas salvas em {save_dir / 'metrics_comparison.png'}")
    
    # 3. FAR vs FRR
    if 'FAR' in metrics and 'FRR' in metrics:
        plt.figure(figsize=(8, 6))
        
        categories = ['FAR\n(False Accept)', 'FRR\n(False Reject)']
        values = [metrics['FAR'], metrics['FRR']]
        colors_far_frr = ['#e74c3c', '#f39c12']
        
        bars = plt.bar(categories, values, color=colors_far_frr, alpha=0.7)
        
        for bar, value in zip(bars, values):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height,
                    f'{value:.3f}',
                    ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        plt.ylim(0, max(values) * 1.2)
        plt.ylabel('Error Rate')
        plt.title('Biometric Error Rates')
        plt.grid(axis='y', alpha=0.3)
        plt.tight_layout()
        plt.savefig(save_dir / 'error_rates.png', dpi=300)
        plt.close()
        logger.info(f"Error rates salvos em {save_dir / 'error_rates.png'}")


def plot_feature_importance(model, feature_names, save_dir: Path, top_n=20):
    """
    Plota importância das features (Random Forest).
    
    Args:
        model: Modelo treinado
        feature_names: Lista de nomes das features
        save_dir: Diretório para salvar
        top_n: Número de features mais importantes para mostrar
    """
    importance = model.get_feature_importance()
    
    if importance is None:
        logger.warning("Modelo não suporta feature importance")
        return
    
    # Ordenar por importância
    indices = np.argsort(importance)[::-1][:top_n]
    top_features = [feature_names[i] for i in indices]
    top_importance = importance[indices]
    
    # Plot
    plt.figure(figsize=(10, 8))
    plt.barh(range(len(top_features)), top_importance, color='steelblue', alpha=0.7)
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel('Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig(save_dir / 'feature_importance.png', dpi=300)
    plt.close()
    
    logger.info(f"Feature importance salvo em {save_dir / 'feature_importance.png'}")


def main():
    parser = argparse.ArgumentParser(description="Treinar classificador anti-spoofing")
    parser.add_argument('--data_dir', type=str, default='data',
                       help='Diretório base com train/val/test')
    parser.add_argument('--model_type', type=str, default='random_forest',
                       choices=['random_forest', 'svm'],
                       help='Tipo de classificador')
    parser.add_argument('--use_fft', action='store_true',
                       help='Usar features de frequência (FFT)')
    parser.add_argument('--optimize', action='store_true',
                       help='Otimizar hiperparâmetros com GridSearch')
    parser.add_argument('--output_model', type=str, default='models/trained_model.pkl',
                       help='Caminho para salvar modelo')
    parser.add_argument('--results_dir', type=str, default='results',
                       help='Diretório para salvar resultados')
    
    args = parser.parse_args()
    
    # Paths
    data_dir = Path(args.data_dir)
    output_model = Path(args.output_model)  # Sempre sobrescreve (último treinado)
    
    # Results separados por tipo de modelo
    results_dir = Path(args.results_dir) / args.model_type
    
    output_model.parent.mkdir(parents=True, exist_ok=True)
    results_dir.mkdir(parents=True, exist_ok=True)
    
    logger.info("=== TREINAMENTO DO MODELO ANTI-SPOOFING ===")
    logger.info(f"Tipo de modelo: {args.model_type}")
    logger.info(f"Usar FFT: {args.use_fft}")
    logger.info(f"Otimizar hiperparâmetros: {args.optimize}")
    
    # Inicializar componentes
    face_detector = FaceDetector()
    feature_extractor = FeatureExtractor(use_fft=args.use_fft)
    
    # === ETAPA 1: Carregar dados ===
    logger.info("\n=== ETAPA 1: Carregando Dados ===")
    
    # Train
    train_real_dir = data_dir / 'train' / 'real'
    train_fake_dir = data_dir / 'train' / 'fake'
    
    if not train_real_dir.exists() or not train_fake_dir.exists():
        logger.error(f"Diretórios de treino não encontrados!")
        logger.error(f"Esperado: {train_real_dir} e {train_fake_dir}")
        logger.error("\nExecute primeiro: python scripts/data_preparation.py")
        return
    
    X_real, y_real = load_images_and_extract_features(
        train_real_dir, label=1, 
        feature_extractor=feature_extractor,
        face_detector=face_detector
    )
    
    X_fake, y_fake = load_images_and_extract_features(
        train_fake_dir, label=0,
        feature_extractor=feature_extractor,
        face_detector=face_detector
    )
    
    # Combinar
    X_train = np.vstack([X_real, X_fake])
    y_train = np.hstack([y_real, y_fake])
    
    logger.info(f"Dataset de treino: {X_train.shape}")
    logger.info(f"  Real: {len(y_real)}, Fake: {len(y_fake)}")
    logger.info(f"  Features por imagem: {X_train.shape[1]}")
    
    # Validação (se existir)
    val_real_dir = data_dir / 'val' / 'real'
    val_fake_dir = data_dir / 'val' / 'fake'
    
    X_val, y_val = None, None
    if val_real_dir.exists() and val_fake_dir.exists():
        X_val_real, y_val_real = load_images_and_extract_features(
            val_real_dir, label=1,
            feature_extractor=feature_extractor,
            face_detector=face_detector
        )
        X_val_fake, y_val_fake = load_images_and_extract_features(
            val_fake_dir, label=0,
            feature_extractor=feature_extractor,
            face_detector=face_detector
        )
        X_val = np.vstack([X_val_real, X_val_fake])
        y_val = np.hstack([y_val_real, y_val_fake])
        logger.info(f"Dataset de validação: {X_val.shape}")
    
    # === ETAPA 2: Treinar modelo ===
    logger.info("\n=== ETAPA 2: Treinando Modelo ===")
    
    model = FaceAntiSpoofing(
        classifier_type=args.model_type,
        use_fft=args.use_fft
    )
    
    train_metrics = model.train(
        X_train, y_train,
        X_val, y_val,
        optimize_hyperparams=args.optimize
    )
    
    logger.info("\nMétricas de Treinamento:")
    for key, value in train_metrics.items():
        logger.info(f"  {key}: {value:.4f}")
    
    # === ETAPA 3: Avaliar no conjunto de teste ===
    test_real_dir = data_dir / 'test' / 'real'
    test_fake_dir = data_dir / 'test' / 'fake'
    
    if test_real_dir.exists() and test_fake_dir.exists():
        logger.info("\n=== ETAPA 3: Avaliando no Conjunto de Teste ===")
        
        X_test_real, y_test_real = load_images_and_extract_features(
            test_real_dir, label=1,
            feature_extractor=feature_extractor,
            face_detector=face_detector
        )
        X_test_fake, y_test_fake = load_images_and_extract_features(
            test_fake_dir, label=0,
            feature_extractor=feature_extractor,
            face_detector=face_detector
        )
        
        X_test = np.vstack([X_test_real, X_test_fake])
        y_test = np.hstack([y_test_real, y_test_fake])
        
        logger.info(f"Dataset de teste: {X_test.shape}")
        
        test_metrics = model.evaluate(X_test, y_test)
        
        logger.info("\n=== RESULTADOS FINAIS ===")
        logger.info(f"Accuracy: {test_metrics['accuracy']:.4f}")
        logger.info(f"Precision: {test_metrics['precision']:.4f}")
        logger.info(f"Recall: {test_metrics['recall']:.4f}")
        logger.info(f"F1-Score: {test_metrics['f1_score']:.4f}")
        logger.info(f"ROC-AUC: {test_metrics['roc_auc']:.4f}")
        logger.info(f"FAR: {test_metrics['FAR']:.4f}")
        logger.info(f"FRR: {test_metrics['FRR']:.4f}")
        
        logger.info("\nClassification Report:")
        print(test_metrics['classification_report'])
        
        # Plotar resultados
        plot_training_results(test_metrics, results_dir)
        
        # Salvar métricas em JSON
        metrics_to_save = {k: v for k, v in test_metrics.items() 
                          if k != 'classification_report'}
        with open(results_dir / 'metrics.json', 'w') as f:
            json.dump(metrics_to_save, f, indent=2)
        logger.info(f"Métricas salvas em {results_dir / 'metrics.json'}")
    
    # === ETAPA 4: Feature Importance (Random Forest) ===
    if args.model_type == 'random_forest':
        logger.info("\n=== ETAPA 4: Analisando Feature Importance ===")
        feature_names = feature_extractor.get_feature_names()
        plot_feature_importance(model, feature_names, results_dir)
    
    # === ETAPA 5: Salvar modelo ===
    logger.info(f"\n=== ETAPA 5: Salvando Modelo ===")
    model.save_model(str(output_model))
    logger.info(f"Modelo salvo em: {output_model}")
    
    # Cleanup
    face_detector.close()
    
    logger.info("\n✅ Treinamento concluído com sucesso!")
    logger.info(f"\nPara usar o modelo:")
    logger.info(f"  streamlit run src/app.py")


if __name__ == "__main__":
    main()
