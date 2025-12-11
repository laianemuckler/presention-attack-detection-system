# 🔐 Sistema de Detecção de Ataques de Apresentação Facial

## 📋 Descrição do Projeto

Sistema inteligente para detectar ataques de apresentação (presentation attacks) em sistemas de verificação biométrica facial, identificando tentativas de fraude usando fotos impressas através da análise de bordas artificiais e texturas.

**Disciplina:** Sistemas Inteligentes Aplicados  
**Problema:** Vulnerabilidade de sistemas biométricos faciais a fotos impressas  
**Solução:** Classificador baseado em ML clássico que detecta bordas artificiais

---

## 🎯 Objetivo

Desenvolver um MVP funcional que:

- ✅ Detecte automaticamente faces em imagens
- ✅ Extraia features de bordas artificiais, textura e frequência
- ✅ Classifique imagens como "Real" ou "Fake" (foto impressa)
- ✅ Forneça interface web intuitiva com visualizações
- ✅ Apresente score de confiança da detecção

---

## 🏗️ Arquitetura do Sistema

```
Input (Imagem Facial)
    ↓
Detecção de Face (MediaPipe/dlib)
    ↓
Extração de Features:
  - Bordas (Canny/Sobel)
  - Textura (LBP)
  - Sharpness (Laplacian)
  - [Opcional] Análise de Frequência (FFT)
    ↓
Classificador ML (Random Forest / SVM)
    ↓
Output: Real/Fake + Confiança
```

---

## 📂 Estrutura do Projeto

```
presentation-attack/
├── src/
│   ├── __init__.py
│   ├── face_detector.py       # Detecção e alinhamento facial
│   ├── feature_extraction.py  # Extração de features (bordas, LBP, etc)
│   ├── classifier.py          # Modelos ML (Random Forest, SVM)
│   ├── utils.py               # Funções auxiliares
│   └── app.py                 # Interface Streamlit
├── data/
│   ├── raw/                   # Dados originais (separados em fake e real)
│   ├── processed/             # Dados processados
│   ├── train/
│   │   ├── real/
│   │   └── fake/
│   └── test/
│       ├── real/
│       └── fake/
├── models/
│   └── trained_model.pkl      # Modelo treinado
├── notebooks/
│   └── exploratory_analysis.ipynb
├── docs/
│   ├── relatorio_tecnico.md
│   └── referencias.md
├── scripts/
│   ├── data_preparation.py
│   └── train.py
├── tests/
│   └── test_classifier.py
├── requirements.txt
└── README.md
```

---

## 🚀 Instalação

### 1. Clonar o repositório

```bash
cd presentation-attack
```

### 2. Criar ambiente virtual (recomendado)

```bash
python -m venv venv
source venv/bin/activate  # Linux/Mac
# ou
venv\Scripts\activate  # Windows
```

### 3. Instalar dependências

```bash
pip install -r requirements.txt
```

---

## 💻 Como Usar

### Treinamento do Modelo

```bash
# 1. Preparar dados
python scripts/data_preparation.py --dataset_path data/raw --output_path data/processed

# 2. Treinar classificador
python scripts/train.py --data_path data/processed --output_model models/trained_model.pkl
```

### Executar Aplicação Web

```bash
streamlit run src/app.py
```

Acesse: `http://localhost:8501`


---

## 🔬 Técnicas de IA Utilizadas

### 1. **Processamento de Imagem (OpenCV)**

- Detecção de bordas: Canny, Sobel, Laplacian
- Análise de sharpness e nitidez
- Transformada de Fourier (análise de frequência)

### 2. **Extração de Features**

- **LBP (Local Binary Patterns)**: Captura micropadrões de textura
- **Densidade de Bordas**: Identifica bordas artificiais em regiões periféricas
- **Variância Laplaciana**: Mede sharpness e qualidade

### 3. **Machine Learning Clássico**

- **Random Forest**: Ensemble de árvores de decisão
- **SVM**: Support Vector Machine com kernel RBF
- Validação cruzada e otimização de hiperparâmetros

---

## 📊 Datasets

### Datasets Públicos Recomendados:

1. **NUAA Photograph Imposter Database** (~10k imagens)
2. **CASIA-FASD** (~600 vídeos)
3. **Replay-Attack Database**

### Criação de Dataset Próprio:

- Coletar 100-200 fotos reais
- Imprimir e fotografar as mesmas imagens
- Variar iluminação, distância e qualidade

---

## 📈 Métricas de Avaliação

- **Accuracy**: Precisão geral
- **Precision/Recall**: Balanceamento entre falsos positivos/negativos
- **F1-Score**: Média harmônica
- **ROC-AUC**: Curva de performance
- **FAR/FRR**: Taxa de falsos aceites/rejeições


## 🔧 Tecnologias

- **Python 3.8+**
- **OpenCV**: Processamento de imagem
- **scikit-learn**: Machine Learning
- **MediaPipe**: Detecção facial
- **Streamlit**: Interface web
- **NumPy, Pandas, Matplotlib**: Análise de dados

---

## 🎓 Ferramenta Interativa

Este projeto faz parte da disciplina **Sistemas Inteligentes Aplicados** e segue as 7 etapas propostas:

### Foto real:
Foto com análise de features:
![foto-real](assets/images/real-photo.png)

Resultado da Análise:
![resultado-da-analise](assets/images/results-real-photo.pngng)

### Foto de tentativa de ataque de apresentação:
Foto com análise de features:
![foto-de-ataque-apresentacao](assets/images/photo-presentation.png)

Resultado da Análise:
![resultado-da-analise](assets/images/results-photo-presentation.png)


---

## 🚧 Trabalhos Futuros

- [ ] Implementar CNN para comparação de performance
- [ ] Adicionar detecção de ataques em vídeo (análise temporal)
- [ ] Suporte a máscaras 3D e ataques de replay
- [ ] Otimização para processamento em tempo real
- [ ] Deploy em produção (API REST)

