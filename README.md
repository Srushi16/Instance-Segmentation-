# 🔬 Instance Segmentation of CT weld Scan of Stator Windings using Deep Learning

This project implements a **Instance image segmentation pipeline** for detecting and classifying defects such as **Pores, Open Pores, and Weld regions** from cross-sectional images using deep learning.

It leverages **PyTorch** and **segmentation_models_pytorch (SMP)** with multiple encoder backbones and includes **training, evaluation, hyperparameter optimization, and visualization tools**.

---

## 🚀 Features

- ✅ End-to-end segmentation pipeline  
- ✅ Custom PyTorch dataset for image-mask handling  
- ✅ Multi-class segmentation (Pores, Open Pores, Weld)  
- ✅ UNet architecture with multiple encoders (VGG, ResNet variants)  
- ✅ Custom loss function (Dice + CrossEntropy)  
- ✅ IoU and Dice score evaluation  
- ✅ Hyperparameter tuning with Optuna  
- ✅ Extensive visualization tools (IoU plots, predictions, etc.)  
- ✅ Class imbalance handling via dynamic class weights  

---

## 🧠 Model Architecture

- **Model:** U-Net  
- **Library:** `segmentation_models_pytorch`  
- **Encoders Tested:**
  - VGG11  
  - ResNet18 / 34 / 50 / 101 / 152  

- **Input Size:** Resized (e.g., 512×512)  
- **Output:** Multi-channel segmentation masks  

---

## 📂 Project Structure
├── segmentation_pipeline.py # Main training & evaluation script
├── README.md # Project documentation
├── requirements.txt # Dependencies
├── data/ # (Not included) Dataset directory
│ ├── images/
│ └── masks/
├── outputs/ # Saved models, plots, logs
└── notebooks/ # (Optional) experiments/visualization


---

## 📊 Dataset

The dataset consists of **cross-sectional images** with corresponding segmentation masks.

### Classes:
- `Pores`
- `Open Pores`
- `Weld`

> ⚠️ **Note:** Dataset is not included in this repository due to size/privacy.  
You must provide your own dataset and update file paths in the script.

---

## ⚙️ Installation

Clone the repository:

```bash
git clone https://github.com/your-username/segmentation-project.git
cd segmentation-project
```
Install dependencies:
```bash
pip install -r requirements.txt
```
---

🛠️ Configuration

Before running, update all file paths in the script:

# Example
```bash
TRAIN_KOP_DIR = "YOUR_PATH_HERE"
MASK_KOP_DIR = "YOUR_PATH_HERE"
```

Also update:
```bash
BEST_MODEL_DIR = "YOUR_OUTPUT_DIRECTORY"
```
---

▶️ Usage

Run the main pipeline:
```bash
python segmentation_pipeline.py
```
This will:

    Load and preprocess datasets
    Train segmentation models
    Evaluate performance (IoU, Dice)
    Save best models
    Generate visualizations

---

📈 Evaluation Metrics

The model is evaluated using:

    IoU (Intersection over Union)
    Dice Loss / Dice Score
    Per-class performance metrics

---

🔍 Hyperparameter Optimization

This project uses Optuna for tuning:

    Learning rate
    Weight decay
    Encoder selection

Results are stored and best models are automatically saved.

---

📊 Visualizations

The pipeline generates:

    📉 Training vs Validation loss curves
    📊 IoU score plots (per class & encoder)
    📦 Boxplots for performance comparison
    🖼️ Segmentation predictions on test images

---

⚠️ Important Notes
    Ensure consistent image-mask naming
    Masks must be properly labeled using defined pixel values
    Class imbalance is handled using computed class weights
    Large datasets may require GPU for training

---

📌 Future Improvements
    Modularize code (train.py, dataset.py, utils.py)
    Add inference script
    Docker support
    Web demo for visualization
    Model export (ONNX / TorchScript)

--- 

👨‍💻 Author

Your Name
📧 rushikesh.sonawane16598@gmail.com
🔗 LinkedIn: https://www.linkedin.com/in/rushikesh-sonawane2025

---

⭐ Acknowledgements
    segmentation_models_pytorch
    PyTorch
    Optuna
    FAPS LAB, FAU ERLANGEN NUREMBERG
