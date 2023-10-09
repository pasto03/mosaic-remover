"""
config for model
"""
import torch


device = 'cuda' if torch.cuda.is_available() else 'cpu'
IMAGE_SIZE = 256
mosaic_window_size = (9, 9)

