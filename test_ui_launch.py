import gradio as gr
import torch
from src.model import SimpleCNN
import os

print("Imports successful")
model = SimpleCNN()
if os.path.exists('cifar_net.pth'):
    model.load_state_dict(torch.load('cifar_net.pth', map_location='cpu'))
    print("Model loaded")
else:
    print("Model not found")
