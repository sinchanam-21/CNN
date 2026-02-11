import gradio as gr
from src.ui import create_ui
import torch

# Use CPU for deployment environment usually, or check CUDA
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

if __name__ == "__main__":
    demo = create_ui(model_path='cifar_net.pth', device=device)
    demo.launch(server_name="0.0.0.0", server_port=7860)
