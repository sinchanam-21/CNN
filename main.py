import torch
import torch.nn as nn
import torch.optim as optim
import argparse
import sys
import os

from src.data_loader import get_dataloaders
from src.model import SimpleCNN
from src.train import train_model
from src.evaluate import evaluate_model
from src.inference import predict_image
from src.utils import plot_history, plot_confusion_matrix
from src.ui import launch_ui

def main():
    parser = argparse.ArgumentParser(description='CIFAR-10 CNN Classifier')
    parser.add_argument('--mode', type=str, default='train', choices=['train', 'predict', 'ui'],
                        help='Mode: train, predict, or ui')
    
    parser.add_argument('--epochs', type=int, default=10, help='Number of epochs to train')
    parser.add_argument('--batch_size', type=int, default=4, help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001, help='Learning rate')
    parser.add_argument('--image', type=str, help='Path to image for prediction')
    parser.add_argument('--model_path', type=str, default='cifar_net.pth', help='Path to saved model')
    parser.add_argument('--plot', action='store_true', help='Plot training history and confusion matrix')
    
    args = parser.parse_args()
    
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    classes = ('plane', 'car', 'bird', 'cat',
               'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    if args.mode == 'train':
        trainloader, testloader, _ = get_dataloaders(batch_size=args.batch_size)
        model = SimpleCNN()
        criterion = nn.CrossEntropyLoss()
        optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=0.9)
        # Add a learning rate scheduler
        scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.1)
        
        history = train_model(model, trainloader, testloader, criterion, optimizer, scheduler=scheduler, num_epochs=args.epochs, device=device)
        
        if args.plot:
            plot_history(history)
            # Final evaluation for Confusion Matrix
            _, y_pred, y_true = evaluate_model(model, testloader, device=device)
            plot_confusion_matrix(y_true, y_pred, classes)
        
    elif args.mode == 'predict':
        if not args.image:
            print("Error: --image argument is required for prediction mode")
            return

        model = SimpleCNN()
        if os.path.exists(args.model_path):
            model.load_state_dict(torch.load(args.model_path, map_location=device))
            print(f"Loaded model from {args.model_path}")
        else:
            print(f"Error: Model file {args.model_path} not found. Train the model first.")
            return

        predict_image(model, args.image, classes, device=device)
        
    elif args.mode == 'ui':
        launch_ui(model_path=args.model_path, device=device)


if __name__ == '__main__':
    main()
