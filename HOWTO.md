# How to Run UPIQAL

## Prerequisites

- Python 3.9 or later
- Git

## 1. Clone the repository

```bash
git clone <repository-url>
cd FR-IQA-Algo
```

## 2. Create and activate a virtual environment

```bash
python3 -m venv venv
```

Activate it:

- **macOS / Linux:**
  ```bash
  source venv/bin/activate
  ```
- **Windows:**
  ```bash
  venv\Scripts\activate
  ```

Your prompt should now show `(venv)`.

## 3. Install dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt
```

### Core dependencies

| Package | Purpose |
|---|---|
| `torch` + `torchvision` | VGG16 backbone, GPU-accelerated tensor ops |
| `numpy` | Array math and windowing functions |
| `scipy` | Cholesky decomposition, wavelet transforms |
| `scikit-image` | Multi-scale pyramid, Canny edge detector, morphological ops |
| `Pillow` | Image I/O (PNG, JPEG, TIFF …) |
| `opencv-python` | Convolutional filters, morphological dilation |
| `POT` | Sinkhorn–Knopp optimal transport (EMD approximation) |
| `colour-science` | sRGB → Oklab color space conversion |

## 4. (Optional) GPU setup

UPIQAL runs on CPU by default. For GPU acceleration install the CUDA-enabled build of PyTorch:

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
```

Verify CUDA is available:

```python
import torch
print(torch.cuda.is_available())
```

## 5. Run the quality assessment

```bash
python main.py --reference path/to/reference.png --target path/to/target.png
```

### CLI options

| Flag | Default | Description |
|---|---|---|
| `--reference` | required | Path to the reference (pristine) image |
| `--target` | required | Path to the distorted / generated image |
| `--output-dir` | `./results` | Directory for diagnostic heatmaps |
| `--device` | `cpu` | `cpu` or `cuda` |
| `--scale` | `3` | Number of Gaussian pyramid levels |

### Example

```bash
python main.py \
  --reference data/ref/img001.png \
  --target data/dist/img001_jpeg.png \
  --output-dir results/ \
  --device cuda
```

## 6. Outputs

The script writes the following to `--output-dir`:

```
results/
├── score.txt              # Scalar FR-IQA score [0, 1]  (higher = better quality)
├── anomaly_map.png        # Global anomaly heatmap (Mahalanobis distance)
├── color_degradation.png  # Chromatic transport degradation map
├── blocking_mask.png      # JPEG blocking artifact mask
├── ringing_mask.png       # Gibbs ringing mask
├── blur_mask.png          # Blur / edge-spread mask
└── noise_mask.png         # Gaussian noise mask
```

## 7. Deactivate the virtual environment

```bash
deactivate
```

## Troubleshooting

**`ModuleNotFoundError: No module named 'torch'`**
Make sure the virtual environment is active (`source venv/bin/activate`) before running any script.

**`CUDA out of memory`**
Reduce the image resolution or use `--device cpu`.

**`colour` package not found**
Install it explicitly: `pip install colour-science`.
