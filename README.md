# CIFAR-10 CNN Classifier

This project implements an end-to-end Convolutional Neural Network (CNN) for the CIFAR-10 dataset using PyTorch.

## Features
- **Data Loading**: Automatically downloads and processes CIFAR-10 data (with SSL fix).
- **Model**: Custom 3-layer CNN architecture.
- **Training**: Configurable training loop.
- **Inference**: Predict classes for new images.

## Requirements
- Python 3.8+
- PyTorch (CPU-only version recommended for Windows to avoid DLL errors)
- Torchvision
- Matplotlib
- Numpy
- Pillow

Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Training
To train the model:
```bash
python main.py --mode train --epochs 10 --batch_size 4
```

### Prediction
To run inference on an image via CLI:
```bash
python main.py --mode predict --image <path_to_image>
```

### Web UI
To launch the interactive Web interface:
```bash
python main.py --mode ui
```
*Gradio will provide a local URL to access the UI.*


## Troubleshooting (Windows)
If you encounter errors:
- **DLL Load Failed**: Ensure [Visual C++ Redistributable](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist?view=msvc-170) is installed. Installing CPU-only PyTorch often helps:
  `pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu`
- **shm.dll Error**: This is a multiprocessing issue. The code handles this by setting `num_workers=0` in `data_loader.py`.
- **SSL Error during download**: The code includes a patch to bypass SSL verification if certificates are missing.
