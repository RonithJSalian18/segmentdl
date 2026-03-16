# Construction Site Segmentation using U-Net

## Overview

This project implements a **semantic segmentation pipeline** for detecting construction sites in aerial imagery using a **U-Net deep learning model** built with TensorFlow.

The pipeline performs:

* Dataset extraction from ZIP files
* Image–mask pairing and validation
* Image preprocessing and normalization
* Data augmentation
* Dataset splitting (Train / Validation / Test)
* Model training using U-Net
* Evaluation with segmentation metrics
* Saving trained models and training logs

The system is designed to run **locally or on a remote server** and can also be containerized using Docker.

---

# Project Structure

```
project/
│
├── server_notebook.py
├── requirements.txt
├── README.md
│
├── data/
│   └── input/
│        ├── original_images.zip
│        └── mask_images.zip
│
└── segmentation_results/
     ├── train/
     ├── val/
     ├── test/
     ├── models/
     ├── logs/
     └── metadata/
```

---

# Dataset Format

The system expects two ZIP files:

### 1. Original Images

```
original_images.zip
```

Contains aerial RGB images.

### 2. Mask Images

```
mask_images.zip
```

Contains segmentation masks.

### Important

Each mask file must have the **same filename as the corresponding image**.

Example:

```
image_001.png
image_001.png (mask)
```

---

# Installation

Clone the repository:

```bash
git clone https://github.com/yourusername/construction-site-segmentation.git
cd construction-site-segmentation
```

Install dependencies:

```bash
pip install -r requirements.txt
```

---

# Required Libraries

The project uses the following libraries:

* TensorFlow
* NumPy
* OpenCV
* Albumentations
* Scikit-learn
* tqdm
* Pillow

All dependencies are included in `requirements.txt`.

---

# Running the Training Pipeline

Place the dataset ZIP files inside:

```
data/input/
```

Then run:

```bash
python server_notebook.py
```

---

# Processing Pipeline

The training script performs the following steps:

1. Extract dataset ZIP files
2. Find matching image–mask pairs
3. Validate dataset
4. Resize images to **512 × 512**
5. Normalize images and masks
6. Apply data augmentation
7. Split dataset into:

   * Train (75%)
   * Validation (15%)
   * Test (10%)
8. Train the **U-Net segmentation model**
9. Evaluate the model
10. Save model and logs

---

# Model Architecture

The project uses a **U-Net convolutional neural network** consisting of:

Encoder:

* Convolution blocks
* MaxPooling layers

Bottleneck:

* Deep convolution layers

Decoder:

* Transposed convolutions
* Skip connections

Output:

* Sigmoid activation for binary segmentation

Input size:

```
512 × 512 × 3
```

---

# Training

Default training configuration:

| Parameter  | Value               |
| ---------- | ------------------- |
| Epochs     | 50                  |
| Batch Size | 32                  |
| Optimizer  | Adam                |
| Loss       | Binary Crossentropy |

Metrics:

* Accuracy
* IoU
* Precision
* Recall

---

# Output

After training, results are stored in:

```
segmentation_results/
```

Contents:

```
models/
construction_site_segmentation.h5
```

```
logs/
training_history.json
```

```
train/
val/
test/
```

```
metadata/
dataset_info.json
```

---

# Evaluation Metrics

The model reports:

* Loss
* Binary Accuracy
* Intersection over Union (IoU)
* Precision
* Recall

---

# Using Docker (Optional)

Build the Docker image:

```bash
docker build -t segmentation-training .
```

Run the container:

```bash
docker run \
-v /server/data:/app/data/input \
-v /server/output:/app/segmentation_results \
segmentation-training
```

---

# Download Results

After training, download the folder:

```
segmentation_results/
```

This contains:

* Trained model
* Processed dataset
* Training logs
* Metadata

---

# Future Improvements

Possible extensions:

* Multi-class segmentation
* Attention U-Net
* Mixed precision training
* GPU optimization
* Model inference pipeline
* Deployment API

---

# License

This project is open-source and available under the **MIT License**.

---

# Author

Developed for **Construction Site Segmentation using Deep Learning**.

If you find this project useful, consider giving it a ⭐ on GitHub.
