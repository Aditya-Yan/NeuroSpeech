# Import model classes from models.py
from .models import GRU, HMRNN

# Make both models available when importing neuralDecoder.models
__all__ = ['GRU', 'HMRNN']
