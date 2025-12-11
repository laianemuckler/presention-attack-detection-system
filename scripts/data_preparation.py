"""
Script de Preparação de Dados
Processa imagens brutas e prepara dataset para treinamento
"""

import cv2
import numpy as np
import argparse
from pathlib import Path
from tqdm import tqdm
import sys
import shutil
import logging

# Adicionar src ao path
sys.path.append(str(Path(__file__).parent.parent / 'src'))

from face_detector import FaceDetector
from feature_extraction import FeatureExtractor

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class DatasetPreparator:
    """
    Prepara dataset para treinamento:
    - Detecta faces nas imagens
    - Extrai features
    - Salva em formato apropriado para treinamento
    """
    
    def __init__(self, target_size=(224, 224)):
        self.target_size = target_size
        self.face_detector = FaceDetector()
        self.feature_extractor = FeatureExtractor()
    
    def process_directory(self, input_dir: Path, output_dir: Path, label: str):
        """
        Processa todas as imagens de um diretório.
        
        Args:
            input_dir: Diretório com imagens originais
            output_dir: Diretório para salvar imagens processadas
            label: 'real' ou 'fake'
        """
        logger.info(f"Processando {label} images de {input_dir}")
        
        # Criar diretório de saída
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Listar imagens
        image_extensions = ['*.jpg', '*.jpeg', '*.png', '*.JPG', '*.JPEG', '*.PNG']
        image_files = []
        for ext in image_extensions:
            image_files.extend(input_dir.glob(ext))
        
        logger.info(f"Encontradas {len(image_files)} imagens")
        
        # Estatísticas
        processed = 0
        failed = 0
        no_face = 0
        
        # Processar cada imagem
        for img_path in tqdm(image_files, desc=f"Processing {label}"):
            try:
                # Carregar imagem
                image = cv2.imread(str(img_path))
                if image is None:
                    logger.warning(f"Não foi possível carregar: {img_path}")
                    failed += 1
                    continue
                
                # Detectar face
                face_result = self.face_detector.detect_face(image)
                
                if face_result is None:
                    logger.warning(f"Nenhuma face detectada em: {img_path.name}")
                    no_face += 1
                    continue
                
                face_roi, face_info = face_result
                
                # Validar qualidade
                is_valid, reason = self.face_detector.validate_face_quality(face_roi)
                if not is_valid:
                    logger.warning(f"Qualidade baixa ({reason}): {img_path.name}")
                    # Ainda assim processar, mas registrar
                
                # Preprocessar
                face_processed = self.face_detector.preprocess_face(
                    face_roi, 
                    target_size=self.target_size
                )
                
                # Salvar imagem processada
                output_path = output_dir / f"{img_path.stem}_processed.jpg"
                cv2.imwrite(str(output_path), face_processed)
                
                processed += 1
                
            except Exception as e:
                logger.error(f"Erro processando {img_path}: {e}")
                failed += 1
        
        # Relatório
        logger.info(f"\n=== Relatório {label.upper()} ===")
        logger.info(f"Total de imagens: {len(image_files)}")
        logger.info(f"Processadas com sucesso: {processed}")
        logger.info(f"Sem face detectada: {no_face}")
        logger.info(f"Falharam: {failed}")
        logger.info(f"Taxa de sucesso: {processed/len(image_files)*100:.1f}%\n")
        
        return processed, no_face, failed
    
    def extract_features_from_directory(self, input_dir: Path, output_file: Path, label: int):
        """
        Extrai features de todas as imagens e salva em arquivo .npz.
        
        Args:
            input_dir: Diretório com imagens processadas
            output_file: Arquivo para salvar features
            label: 0 (fake) ou 1 (real)
        """
        logger.info(f"Extraindo features de {input_dir}")
        
        # Listar imagens
        image_files = list(input_dir.glob("*.jpg")) + list(input_dir.glob("*.png"))
        
        features_list = []
        labels_list = []
        filenames_list = []
        
        for img_path in tqdm(image_files, desc="Extracting features"):
            try:
                # Carregar imagem
                image = cv2.imread(str(img_path))
                if image is None:
                    continue
                
                # Extrair features
                features = self.feature_extractor.extract_all_features(image)
                
                features_list.append(features)
                labels_list.append(label)
                filenames_list.append(img_path.name)
                
            except Exception as e:
                logger.error(f"Erro extraindo features de {img_path}: {e}")
        
        # Converter para numpy arrays
        X = np.array(features_list)
        y = np.array(labels_list)
        
        # Salvar
        np.savez_compressed(
            output_file,
            X=X,
            y=y,
            filenames=np.array(filenames_list)
        )
        
        logger.info(f"Features salvas: {X.shape} -> {output_file}")
        return X, y
    
    def split_dataset(self, input_dir: Path, output_dir: Path, 
                     train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
        """
        Divide dataset em train/val/test mantendo distribuição de classes.
        
        Args:
            input_dir: Diretório com imagens processadas (deve ter subpastas real/ e fake/)
            output_dir: Diretório base para salvar splits
            train_ratio: Proporção para treino
            val_ratio: Proporção para validação
            test_ratio: Proporção para teste
        """
        assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 0.001, "Ratios devem somar 1.0"
        
        for label in ['real', 'fake']:
            label_dir = input_dir / label
            if not label_dir.exists():
                logger.warning(f"Diretório não encontrado: {label_dir}")
                continue
            
            # Listar imagens
            images = list(label_dir.glob("*.jpg")) + list(label_dir.glob("*.png"))
            logger.info(f"Dividindo {len(images)} imagens de {label}")
            
            # Embaralhar
            np.random.shuffle(images)
            
            # Calcular índices de split
            n_train = int(len(images) * train_ratio)
            n_val = int(len(images) * val_ratio)
            
            train_images = images[:n_train]
            val_images = images[n_train:n_train+n_val]
            test_images = images[n_train+n_val:]
            
            # Criar diretórios
            for split_name, split_images in [
                ('train', train_images),
                ('val', val_images),
                ('test', test_images)
            ]:
                split_dir = output_dir / split_name / label
                split_dir.mkdir(parents=True, exist_ok=True)
                
                # Copiar imagens
                for img_path in split_images:
                    shutil.copy2(img_path, split_dir / img_path.name)
            
            logger.info(f"  Train: {len(train_images)}, Val: {len(val_images)}, Test: {len(test_images)}")


def main():
    parser = argparse.ArgumentParser(description="Preparar dataset para treinamento")
    parser.add_argument('--raw_dir', type=str, default='data/raw',
                       help='Diretório com imagens originais (deve ter subpastas real/ e fake/)')
    parser.add_argument('--processed_dir', type=str, default='data/processed',
                       help='Diretório para salvar imagens processadas')
    parser.add_argument('--output_dir', type=str, default='data',
                       help='Diretório base para train/val/test splits')
    parser.add_argument('--extract_features', action='store_true',
                       help='Extrair features e salvar em .npz')
    parser.add_argument('--split_only', action='store_true',
                       help='Apenas dividir dataset existente (pular processamento)')
    
    args = parser.parse_args()
    
    # Paths
    raw_dir = Path(args.raw_dir)
    processed_dir = Path(args.processed_dir)
    output_dir = Path(args.output_dir)
    
    # Inicializar preparador
    preparator = DatasetPreparator()
    
    if not args.split_only:
        # Processar imagens
        logger.info("=== ETAPA 1: Processando Imagens ===")
        
        for label in ['real', 'fake']:
            input_path = raw_dir / label
            output_path = processed_dir / label
            
            if not input_path.exists():
                logger.warning(f"Diretório não encontrado: {input_path}")
                logger.info(f"Criando estrutura de exemplo...")
                input_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"Por favor, adicione imagens em: {input_path}")
                continue
            
            preparator.process_directory(input_path, output_path, label)
    
    # Dividir dataset
    logger.info("\n=== ETAPA 2: Dividindo Dataset ===")
    preparator.split_dataset(processed_dir, output_dir)
    
    # Extrair features (opcional)
    if args.extract_features:
        logger.info("\n=== ETAPA 3: Extraindo Features ===")
        
        for split in ['train', 'val', 'test']:
            logger.info(f"\nProcessando split: {split}")
            
            # Real (label=1)
            real_dir = output_dir / split / 'real'
            if real_dir.exists():
                preparator.extract_features_from_directory(
                    real_dir,
                    output_dir / f'{split}_real_features.npz',
                    label=1
                )
            
            # Fake (label=0)
            fake_dir = output_dir / split / 'fake'
            if fake_dir.exists():
                preparator.extract_features_from_directory(
                    fake_dir,
                    output_dir / f'{split}_fake_features.npz',
                    label=0
                )
    
    logger.info("\n✅ Preparação de dados concluída!")
    logger.info(f"Estrutura criada em: {output_dir}")
    logger.info("\nPróximos passos:")
    logger.info("  1. Verificar imagens processadas em data/train/, data/val/, data/test/")
    logger.info("  2. Executar: python scripts/train.py")


if __name__ == "__main__":
    main()
