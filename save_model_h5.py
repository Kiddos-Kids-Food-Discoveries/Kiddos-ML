# Import
import tensorflow as tf
from tensorflow.keras.models import load_model
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# Memuat model dari file .keras
model = load_model('best_model.keras')

# Menyimpan ulang model ke format .h5
model.save('model.h5')