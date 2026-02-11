import gradio as gr
import torch
import torchvision.transforms as transforms
from PIL import Image
from src.model import SimpleCNN
import os

def create_ui(model_path='cifar_net.pth', device='cpu'):
    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')
               
    model = SimpleCNN()
    if os.path.exists(model_path):
        model.load_state_dict(torch.load(model_path, map_location=device))
        print(f"Loaded model from {model_path}")
    else:
        print(f"Error: Model file {model_path} not found.")
        return

    model.to(device)
    model.eval()

    transform = transforms.Compose([
        transforms.Resize((32, 32)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    def predict(img):
        if img is None:
            return None
        img = transform(img).unsqueeze(0).to(device)
        with torch.no_grad():
            outputs = model(img)
            probs = torch.nn.functional.softmax(outputs[0], dim=0)
            confidences = {classes[i]: float(probs[i]) for i in range(10)}
        return confidences

    demo = gr.Interface(
        fn=predict,
        inputs=gr.Image(type="pil"),
        outputs=gr.Label(num_top_classes=3),
        title="CIFAR-10 Image Classifier",
        description="Upload an image to see what the CNN predicts! (Classes: plane, car, bird, cat, deer, dog, frog, horse, ship, truck)",
        theme="soft"
    )
    return demo

def launch_ui(model_path='cifar_net.pth', device='cpu'):
    demo = create_ui(model_path, device)
    demo.launch(share=True)
