---
title: CIFAR 10
emoji: 🔥
colorFrom: yellow
colorTo: purple
sdk: docker
pinned: false
license: mit
---

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

## Deployment

### Docker

To run the application using Docker:

1. **Build the image**:
   ```bash
   docker build -t cnn-classifier .
   ```

2. **Run the container**:
   ```bash
   docker run -p 7860:7860 cnn-classifier
   ```

3. Open `http://localhost:7860` in your browser.

### Hugging Face Spaces

This project is ready for deployment to Hugging Face Spaces (CPU Basic).

1. Create a new Space on [huggingface.co](https://huggingface.co/new-space).
2. Select **Gradio** as the SDK.
3. Upload the files (including `app.py`, `cifar_net.pth`, `requirements.txt`, and the `src/` folder).
4. The Space will build and launch automatically using `app.py`.

Check out the configuration reference at https://huggingface.co/docs/hub/spaces-config-reference
