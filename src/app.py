"""
Aplicação Web Streamlit - Detector Anti-Spoofing Facial
Interface intuitiva para upload, análise e visualização de resultados
"""

import streamlit as st
import cv2
import numpy as np
from PIL import Image
import sys
from pathlib import Path
import plotly.graph_objects as go
import plotly.express as px
import json

# Adicionar diretório src ao path
sys.path.append(str(Path(__file__).parent))

from classifier import FaceAntiSpoofing
from feature_extraction import FeatureExtractor
import utils

# Configuração da página
st.set_page_config(
    page_title="Anti-Spoofing Facial",
    page_icon="🔐",
    layout="wide",
    initial_sidebar_state="expanded"
)

# CSS customizado
st.markdown("""
    <style>
    .main {
        padding: 0rem 1rem;
    }
    .stAlert {
        margin-top: 1rem;
    }
    .metric-card {
        background-color: #f0f2f6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin: 0.5rem 0;
    }
    /* Diminuir tamanho dos números das métricas */
    [data-testid="stMetricValue"] {
        font-size: 1.2rem;
    }
    [data-testid="stMetricLabel"] {
        font-size: 0.9rem;
    }
    </style>
""", unsafe_allow_html=True)


@st.cache_resource
def load_model(model_path):
    """Carrega modelo treinado (com cache)."""
    try:
        detector = FaceAntiSpoofing(model_path=model_path)
        return detector, None
    except Exception as e:
        return None, str(e)


def process_image(uploaded_file):
    """Converte arquivo upado para array numpy."""
    image = Image.open(uploaded_file)
    image = np.array(image)
    # Converter RGB para BGR (OpenCV usa BGR)
    if len(image.shape) == 3 and image.shape[2] == 3:
        image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    return image


def display_metrics(prediction, confidence, details):
    """Exibe métricas de predição."""
    col1, col2, col3 = st.columns(3)
    
    with col1:
        if prediction == 'real':
            st.success("✅ IMAGEM REAL")
        elif prediction == 'fake':
            st.error("⚠️ POSSÍVEL FRAUDE")
        else:
            st.warning("❓ DESCONHECIDO")
    
    with col2:
        st.metric("Confiança", f"{confidence:.2%}")
    
    with col3:
        st.metric("Features Extraídas", details.get('features_extracted', 'N/A'))


def plot_confidence_gauge(confidence, prediction):
    """Cria gráfico de gauge para confiança."""
    color = "green" if prediction == "real" else "red"
    
    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=confidence * 100,
        title={'text': "Nível de Confiança"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 50], 'color': "lightgray"},
                {'range': [50, 75], 'color': "gray"},
                {'range': [75, 100], 'color': "lightgreen" if prediction == "real" else "lightcoral"}
            ],
            'threshold': {
                'line': {'color': "black", 'width': 4},
                'thickness': 0.75,
                'value': 80
            }
        }
    ))
    
    fig.update_layout(height=300)
    return fig


def plot_probability_bars(details):
    """Cria gráfico de barras com probabilidades."""
    prob_real = details.get('probability_real', 0)
    prob_fake = details.get('probability_fake', 0)
    
    fig = go.Figure(data=[
        go.Bar(
            x=['Real', 'Fake'],
            y=[prob_real, prob_fake],
            marker_color=['green', 'red'],
            text=[f'{prob_real:.2%}', f'{prob_fake:.2%}'],
            textposition='auto',
        )
    ])
    
    fig.update_layout(
        title="Probabilidades de Classificação",
        yaxis_title="Probabilidade",
        yaxis=dict(range=[0, 1]),
        height=300
    )
    
    return fig


def main():
    # Sidebar
    with st.sidebar:
        st.image("https://img.icons8.com/color/96/000000/face-id.png", width=80)
        st.title("🔐 Anti-Spoofing")
        st.markdown("---")
        
        st.subheader("Configurações")
        
        # Caminho do modelo
        model_path = st.text_input(
            "Caminho do Modelo",
            value="models/trained_model.pkl",
            help="Caminho para o modelo .pkl treinado"
        )
        
        # Opções de visualização
        show_edge_detection = st.checkbox("Mostrar Detecção de Bordas", value=True)
        show_face_bbox = st.checkbox("Mostrar Bounding Box da Face", value=True)
        show_detailed_metrics = st.checkbox("Mostrar Métricas Detalhadas", value=False)
        
        st.markdown("---")
        st.subheader("Sobre o Sistema")
        st.markdown("""
        Sistema de detecção de ataques de apresentação usando:
        - 🔍 Análise de bordas artificiais
        - 🧩 Local Binary Patterns (LBP)
        - 📊 Machine Learning (Random Forest/SVM)
        
        **Como usar:**
        1. Faça upload de uma imagem facial
        2. O sistema detecta automaticamente a face
        3. Análise as features extraídas
        4. Veja o resultado da classificação
        """)
        
        # Métricas do modelo treinado
        st.markdown("---")
        st.subheader("📈 Métricas do Modelo")
        metrics_path = Path("results/metrics.json")
        if metrics_path.exists():
            try:
                with open(metrics_path, 'r') as f:
                    metrics = json.load(f)
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Acurácia", f"{metrics.get('accuracy', 0):.2%}")
                    st.metric("Precisão", f"{metrics.get('precision', 0):.2%}")
                with col2:
                    st.metric("Recall", f"{metrics.get('recall', 0):.2%}")
                    st.metric("F1-Score", f"{metrics.get('f1_score', 0):.2%}")
                
                with st.expander("Taxas de Erro"):
                    st.metric("FAR (False Accept)", f"{metrics.get('FAR', 0):.2%}")
                    st.metric("FRR (False Reject)", f"{metrics.get('FRR', 0):.2%}")
            except Exception as e:
                st.error(f"Erro ao carregar métricas: {e}")
        else:
            st.info("Métricas disponíveis após treinamento")
    
    # Main content
    st.title("🔐 Detector Anti-Spoofing Facial")
    st.markdown("**Identifique tentativas de fraude por foto**")
    
    # Verificar se modelo existe
    if not Path(model_path).exists():
        st.warning(f"⚠️ Modelo não encontrado em: `{model_path}`")
        st.info("""
        **Para treinar um modelo:**
        1. Prepare seu dataset em `data/train/` (pastas `real/` e `fake/`)
        2. Execute: `python scripts/train.py`
        3. O modelo será salvo em `models/trained_model.pkl`
        
        **Ou use o modo de demonstração abaixo** (sem modelo treinado)
        """)
        
        use_demo_mode = st.checkbox("Usar Modo Demonstração (sem predição real)")
    else:
        use_demo_mode = False
    
    # Upload de imagem
    st.markdown("---")
    st.subheader("📤 Upload de Imagem")
    
    uploaded_file = st.file_uploader(
        "Selecione uma imagem facial",
        type=['jpg', 'jpeg', 'png'],
        help="Formatos suportados: JPG, JPEG, PNG"
    )
    
    # Ou usar exemplo
    col1, col2 = st.columns([3, 1])
    with col2:
        use_example = st.button("🎭 Usar Imagem de Exemplo")
    
    if uploaded_file or use_example:
        if use_example:
            # Criar imagem de exemplo sintética
            example_image = np.ones((400, 400, 3), dtype=np.uint8) * 220
            cv2.circle(example_image, (200, 200), 80, (180, 150, 120), -1)
            cv2.circle(example_image, (175, 180), 12, (50, 50, 50), -1)
            cv2.circle(example_image, (225, 180), 12, (50, 50, 50), -1)
            cv2.ellipse(example_image, (200, 220), (35, 15), 0, 0, 180, (100, 50, 50), -1)
            image = example_image
            st.info("Usando imagem de exemplo sintética")
        else:
            image = process_image(uploaded_file)
        
        # Layout de 2 colunas
        col1, col2 = st.columns(2)
        
        with col1:
            st.subheader("🖼️ Imagem Original")
            display_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            st.image(display_image, use_container_width=True)
        
        with col2:
            st.subheader("🔍 Análise de Bordas")
            
            # Extrair features para visualização
            extractor = FeatureExtractor()
            visualizations = extractor.visualize_edges(image)
            
            tabs = st.tabs(["Canny", "Sobel", "Laplacian", "LBP"])
            
            with tabs[0]:
                st.image(visualizations['canny'], use_container_width=True, caption="Detecção de Bordas (Canny)")
            
            with tabs[1]:
                st.image(visualizations['sobel_combined'], use_container_width=True, caption="Gradientes Sobel")
            
            with tabs[2]:
                st.image(visualizations['laplacian'], use_container_width=True, caption="Laplaciano")
            
            with tabs[3]:
                st.image(visualizations['lbp'], use_container_width=True, caption="Local Binary Patterns")
        
        st.markdown("---")
        
        # Predição
        if not use_demo_mode:
            with st.spinner("🔄 Analisando imagem..."):
                detector, error = load_model(model_path)
                
                if detector is None:
                    st.error(f"❌ Erro ao carregar modelo: {error}")
                else:
                    try:
                        prediction, confidence, details = detector.predict(image)
                        
                        # Exibir resultado principal
                        st.subheader("📊 Resultado da Análise")
                        display_metrics(prediction, confidence, details)
                        
                        # Gráficos
                        col1, col2 = st.columns(2)
                        
                        with col1:
                            fig_gauge = plot_confidence_gauge(confidence, prediction)
                            st.plotly_chart(fig_gauge, use_container_width=True)
                        
                        with col2:
                            fig_bars = plot_probability_bars(details)
                            st.plotly_chart(fig_bars, use_container_width=True)
                        
                        # Detalhes técnicos
                        if show_detailed_metrics:
                            st.subheader("🔬 Detalhes Técnicos")
                            
                            col1, col2, col3 = st.columns(3)
                            
                            with col1:
                                st.metric("Probabilidade Real", f"{details['probability_real']:.4f}")
                            with col2:
                                st.metric("Probabilidade Fake", f"{details['probability_fake']:.4f}")
                            with col3:
                                st.metric("Confiança Face", f"{details['face_confidence']:.4f}")
                            
                            with st.expander("Ver todos os detalhes"):
                                st.json(details)
                        
                        # Interpretação
                        st.subheader("💡 Interpretação")
                        if prediction == 'real':
                            st.success("""
                            ✅ **A imagem foi classificada como REAL**
                            
                            Características detectadas:
                            - Padrões de textura consistentes com pele real
                            - Densidade de bordas dentro do esperado
                            - Ausência de artefatos de impressão
                            """)
                        elif prediction == 'fake':
                            st.error("""
                            ⚠️ **POSSÍVEL TENTATIVA DE FRAUDE DETECTADA**
                            
                            Indicadores de foto impressa:
                            - Bordas artificiais detectadas nas regiões periféricas
                            - Padrões de textura inconsistentes
                            - Possíveis artefatos de impressão (dot patterns)
                            
                            **Recomendação:** Solicitar nova captura ou verificação adicional
                            """)
                        
                    except Exception as e:
                        st.error(f"❌ Erro durante análise: {str(e)}")
                        import logging
                        logging.error(f"Erro na predição: {str(e)}", exc_info=True)
        else:
            # Modo demo
            st.info("""
            **Modo Demonstração Ativo**
            
            Neste modo, você pode visualizar a extração de features e análise de bordas,
            mas a classificação Real/Fake não está disponível sem um modelo treinado.
            
            Execute o treinamento conforme instruções acima para ativar todas as funcionalidades.
            """)
    
    else:
        # Estado inicial
        st.info("👆 Faça upload de uma imagem facial para começar a análise")
        
        # Exemplos visuais
        st.markdown("---")
        st.subheader("📚 Como Funciona")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.markdown("""
            **Detecção de Fotos Impressas:**
            
            1. **Análise de Bordas**: Fotos impressas têm bordas artificiais detectáveis
            2. **Textura (LBP)**: Padrões de impressão diferem da pele real
            3. **Sharpness**: Medição de nitidez e qualidade
            4. **Classificação ML**: Random Forest ou SVM decide
            """)
        
        with col2:
            st.markdown("""
            **Features Extraídas:**
            
            - Densidade de bordas (Canny, Sobel)
            - Local Binary Patterns (59 bins)
            - Variância Laplaciana
            - Gradientes Sobel
            - [Opcional] Análise de frequência (FFT)
            
            **Total:** ~24+ features por imagem
            """)


if __name__ == "__main__":
    main()
