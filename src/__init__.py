"""
Sistema de Detecção de Ataques de Apresentação Facial
Anti-Spoofing com Machine Learning Clássico
"""

__version__ = "1.0.0"
__author__ = "Projeto Final - Sistemas Inteligentes"

from .face_detector import FaceDetector
from .feature_extraction import FeatureExtractor
from .classifier import FaceAntiSpoofing

__all__ = ['FaceDetector', 'FeatureExtractor', 'FaceAntiSpoofing']
