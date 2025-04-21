# Digit Detection and Classification Project

This project implements a digit detection and classification system using PyTorch and OpenCV. It can process images containing digits and output both the detected sequence and visualizations of the detections.

## Prerequisites

- Python 3.11
- CUDA-compatible GPU (recommended for better performance)
- UV package manager (for dependency management)

## Environment Setup

1. Install UV package manager (if not already installed):
   ```bash
   pip install uv
   ```

2. Create and activate a virtual environment:
   ```bash
   # Create virtual environment with Python 3.11
   uv venv --python 3.11.12
   
   # Activate the virtual environment
   # On Windows:
   .\.venv\Scripts\activate
   # On Linux/Mac:
   source .venv/bin/activate
   ```

3. Install dependencies from pyproject.toml:
   ```bash
   uv sync
   ```

## Project Structure

```
.
├── src/                    # Source code
│   ├── components/        # Core components
│   ├── pipeline/         # Training and inference pipelines
│   └── utils.py          # Utility functions
├── images/               # Input images for testing
├── graded_images/       # Processed output images
├── artifacts/           # Training artifacts and data
│   ├── data/           # Training and test datasets
│   │   ├── train/     # Training images
│   │   └── test/      # Test images
│   └── models/        # Trained model checkpoints
├── best_model.pth      # Pre-trained model weights
├── run.py              # Main inference script
└── pyproject.toml      # Project configuration
```

## Usage

1. Place your test images in the `images/` directory with names like `1.png`, `2.png`, etc. (already done)

2. Run the inference script:
   ```bash
   python run.py
   ```

3. The processed images will be saved in the `graded_images/` directory with the same names as the input images.

## Output

For each processed image, you will see:
- A visualization of the detected digits with bounding boxes
- Confidence scores for each detection
- The final detected sequence

The processed images will be saved in the `graded_images/` directory with:
- Green bounding boxes around detected digits
- Confidence scores displayed above each detection
- The final detected sequence displayed at the top of the image

## Training the Model

To train the model with your own dataset:

1. Extract your training and test images to the following directory structure:
   ```
   artifacts/
   └── data/
       ├── train/     # Training images
       │   ├── 0/     # Images of digit 0
       │   ├── 1/     # Images of digit 1
       │   ...
       │   └── 9/     # Images of digit 9
       │   └── 10/    # Images of background classa
       └── test/      # Test images
           ├── 0/     # Images of digit 0
           ├── 1/     # Images of digit 1
           ...
           └── 9/     # Images of digit 9
           └── 10/    # Images of background classa
   ```

2. Open `src/pipeline/train.py` and comment out the models you want to train in the `models` dictionary. For example:
   ```python
   models = {
       'SimpleCNN': SimpleCNN(),
       # 'MediumCNN': MediumCNN(),
       # 'ComplexCNN': ComplexCNN(),
       # 'DenseNet': DenseNet(),
       # 'VGG16FineTune': VGG16FineTune()
   }
   ```

3. Run the training script:
   ```bash
   python -m src.pipeline.train
   ```

The training process will:
- Train the selected models
- Save the best model weights
- Generate training curves and metrics
- Save all results in the `artifacts/models` directory with timestamps

