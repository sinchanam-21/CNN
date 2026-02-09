import torch
import torchvision.transforms as transforms
from PIL import Image

def predict_image(model, image_path, classes, device='cpu'):
    """
    Predicts the class of an image using the trained model.
    """
    model.to(device)
    model.eval()
    
    transform = transforms.Compose(
        [transforms.Resize((32, 32)),
         transforms.ToTensor(),
         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])
    
    try:
        image = Image.open(image_path)
        image = transform(image).unsqueeze(0)
        image = image.to(device)
        
        with torch.no_grad():
            outputs = model(image)
            _, predicted = torch.max(outputs, 1)
            
        print(f'Predicted: {classes[predicted.item()]}')
        return classes[predicted.item()]
    except Exception as e:
        print(f"Error predicting image: {e}")
        return None
