# 🎯 Tutorial Passo a Passo

## Para Começar do Zero

### Passo 1: Setup Inicial (5 min)

```bash
# 1.1. Navegar até o diretório do projeto
cd "\presentation-attack"

# 1.2. Criar ambiente virtual
python -m venv venv

# 1.3. Ativar ambiente virtual (Windows)
venv\Scripts\activate

# 1.4. Instalar dependências
pip install -r requirements.txt
```

**Verificação:**

```bash
python -c "import cv2, sklearn, streamlit; print('✅ Tudo instalado!')"
```

---

### Passo 2: Preparar Dataset 

#### Opção A: Dataset Mínimo para Teste (Recomendado para início)

```bash
# 2.1. Criar estrutura de diretórios
mkdir -p data/raw/real data/raw/fake

# 2.2. Adicionar pelo menos 20 imagens de cada classe
# Real: Selfies de pessoas diferentes
# Fake: Fotos dessas selfies impressas e fotografadas
```

**Dica:** Comece com 20-50 imagens por classe para testes rápidos.

#### Opção B: Download de Dataset Público

**NUAA Photograph Imposter Database:**

1. Acesse: http://www.nuaa.edu.cn/
2. Baixe o dataset
3. Extraia em `data/raw/`

---

### Passo 3: Processar Imagens

```bash
# 3.1. Processar e dividir dataset
python scripts/data_preparation.py

# Saída esperada:
# ✅ Faces detectadas e extraídas
# ✅ Imagens redimensionadas para 224x224
# ✅ Dataset dividido em train/val/test (70/15/15)
```

**Verificar resultado:**

```bash
ls data/train/real/  # Deve ter ~70% das imagens reais
ls data/train/fake/  # Deve ter ~70% das imagens fake
```

---

### Passo 4: Treinar Primeiro Modelo

```bash
# 4.1. Treinar Random Forest (mais rápido)
python scripts/train.py --model_type random_forest

# Saída esperada:
# ✅ Modelo treinado
# ✅ Métricas exibidas
# ✅ Modelo salvo em models/trained_model.pkl
# ✅ Gráficos em results/
```

**Verificar modelo:**

```bash
ls models/trained_model.pkl  # Deve existir
ls results/*.png             # Gráficos gerados
```

---

### Passo 5: Testar Aplicação

```bash
# 5.1. Iniciar aplicação web
streamlit run src/app.py

# 5.2. Abrir navegador em: http://localhost:8501

# 5.3. Fazer upload de uma imagem de teste
# 5.4. Ver resultado da análise
```

---

## Workflow Completo de Desenvolvimento

### Ciclo de Iteração

```
1. Coletar Dados
   ↓
2. Processar (data_preparation.py)
   ↓
3. Treinar (train.py)
   ↓
4. Avaliar Métricas
   ↓
5. Ajustar (se necessário)
   ↓
6. Testar em Aplicação
```

### Melhorando o Modelo

#### Se Accuracy < 80%:

**1. Aumentar Dataset**

```bash
# Adicionar mais imagens em data/raw/
# Re-processar
python scripts/data_preparation.py
```

**2. Otimizar Hiperparâmetros**

```bash
python scripts/train.py --optimize
# Demora mais, mas encontra melhores parâmetros
```

**3. Adicionar Features de Frequência**

```bash
python scripts/train.py --use_fft
# Adiciona análise FFT (mais features)
```


## Testando com Suas Próprias Imagens

### 1. Via Interface Web

```bash
streamlit run src/app.py
# Upload manual na interface
```





