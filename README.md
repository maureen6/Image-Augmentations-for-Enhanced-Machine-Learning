# CIFAR-10 IMAGE AUGMENTATION AND CNN TRAINING

## OVERVIEW

This project demonstrates the use of image augmentation techniques to improve the performance of a Convolutional Neural Network (CNN) trained on the CIFAR-10 dataset. The project compares model performance on the original dataset versus an augmented dataset.

## FEATURES

Image Augmentations: Random horizontal flip, rotation, cropping, color jitter, and Gaussian blur.

CNN Model Training: Trains a simple CNN on both original and augmented datasets.

Performance Evaluation: Compares accuracy and loss curves between models.

Visualizations: Displays augmented images and training progress.

## 📂 DATASET
- **Dataset:** CIFAR-10 (60,000 images across 10 classes)
- **Classes:** Airplane, Automobile, Bird, Cat, Deer, Dog, Frog, Horse, Ship, Truck
- **Size:** 32x32 RGB images
- **Download:** [CIFAR-10 Official Site](https://www.cs.toronto.edu/~kriz/cifar.html)

## 🔄 IMAGE AUGMENTATION TECHNIQUES
The following augmentations are applied using **Torchvision Transforms**:
1. **Random Horizontal Flip**
2. **Random Rotation**
3. **Random Crop**
4. **Color Jitter**
5. **Gaussian Blur**

## 🛠️ TECHNOLOGIES USED: 
- **Python 3.x**
- **PyTorch** (`torch`, `torchvision`)
- **Matplotlib** (for visualization)
- **NumPy** (for data manipulation)

## 📌 PROJECT STRUCTURE

```md
Image-Augmentations-for-Enhanced-Machine-Learning/
│── data/                          # Directory for storing the CIFAR-10 dataset
│── notebook/                      # Jupyter notebooks for visualization
│   ├── assignment.ipynb           # Data exploration and augmentation visualization
│── scripts/                       # Source code directory
│   ├── train.py                   # Data loading & augmentation transformations and Training and Evaluation script
│   ├── model.py                   # CNN model definition
│   ├── utils.py                   # Helper functions for training, evaluation and visualization
│── requirements.txt               # Python dependencies
│── README.md                      # Project documentation

```

## 🚀 INSTALLATION

### 1. Clone the Repository

```sh
https://github.com/matilda1740/Image-Augmentations-for-Enhanced-Machine-Learning.git
```
### 2. Create Virtual Environment (optional) - on Mac
```sh
python -m venv env
source env/bin/activate
```

### 3. Install the Dependencies from requirements.txt File
```sh
pip install -r requirements.txt
```

### 4. Run the Script
```sh
python scripts/train.py
```

## 📊 RESULTS AND ANALYSIS

1. **Augmented models showed better generalization.**
2. **Loss curves indicated reduced overfitting with augmentation.**
3. **Final accuracy comparison:**
    1. **Without Augmentation: ~72%**
    2. **With Augmentation: ~78%**

## 🖼️ VISUALIZATION 

1. **Original Vs Augmented Images**
2. **Training Loss Curves for Evaluation**