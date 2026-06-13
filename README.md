# 🧠 Multiclass Brain MRI Classification Using Deep Convolutional Neural Networks (CNN)

[![TensorFlow](https://img.shields.shields.io/badge/TensorFlow-2.x-orange?logo=tensorflow)](https://tensorflow.org)
[![Framework](https://img.shields.shields.io/badge/Framework-Keras-red?logo=keras)](https://keras.io)

## 📌 Project Overview
Automated neuroimaging evaluation plays a vital role in modern clinical decision-making. This project presents a robust, end-to-end computer vision pipeline designed to classify Brain Magnetic Resonance Imaging (MRI) scans into three distinct clinical categories: **Normal (Healthy), Tumor, and Stroke**. 

Built entirely using **TensorFlow/Keras**, the system leverages a custom-designed Deep Convolutional Neural Network (CNN) architecture featuring strict regularization, normalization layers, and dynamic learning rate scheduling to ensure high generalization and precision across noisy imaging data.

---

## 🔬 Comprehensive Technical Methodology

The architecture of this project is systematically structured into a unified five-phase pipeline:

### 1. Mathematical Data Simulation & Pathological Modeling
To rigorously stress-test the classification limits of the structural convolutional layers, the script implements an advanced clinical image generator function (`create_sample_dataset`) that constructs **2,000 multi-channel (RGB) structural tensors** at a resolution of $128 \times 128 \times 3$:

* **Class 0: Normal Tissue (800 Samples):** Represents typical neuroanatomical structures, modeled using baseline matrices embedded with continuous low-frequency Gaussian noise to simulate the typical artifacts found in actual MRI scanners.
* **Class 1: Brain Tumor (700 Samples):** Simulates localized, high-contrast structural masses. This is achieved by programmatically injecting sharp, geometric, high-intensity circular structures via OpenCV contour functions, testing the model's ability to track abnormal structural borders.
* **Class 2: Ischemic Stroke (500 Samples):** Simulates acute cerebrovascular blockages. The script injects dense, oriented hyper/hypo-intense linear bands that slash through baseline structural tissue, forcing the CNN to learn directional edge features.

### 2. Digital Image Preprocessing & Stratified Splitting
Prior to feeding raw matrices into the neural layers, data is passed through a deterministic preprocessing function (`preprocess_data`):
* **Anatomical Aspect-Ratio Padding:** Standard resizing techniques forcibly stretch or squash medical data, distorting pathological features. To eliminate this, the code implements `cv2.copyMakeBorder` to calculate and append zero-padding dynamically, mapping all irregular matrices into standard geometric inputs.
* **Min-Max Intensity Scale:** Normalizes the pixel arrays to a strict floating-point scale of $[0, 1]$ using the following transformation:
  $$X_{\text{norm}} = \frac{X - X_{\text{min}}}{X_{\text{max}} - X_{\text{min}}}$$
  This operation bounds the tensor dynamics, preventing exploding gradients during backpropagation.
* **Data Stratification:** To ensure an un-biased split under multi-class distribution, data is separated into **Train (70%)**, **Validation (15%)**, and **Test (15%)** sets via `sklearn` validation loops, keeping the precise label proportions identical across all subsets.

### 3. Comprehensive Deep CNN Structural Design
The architecture avoids heavy pre-trained models to eliminate domain-transfer errors, relying instead on a highly tuned, **4-Stage Deep Sequential Convolutional Network**:

| Layer Type | Filter Dimension / Size | Activation | Regularization / Operational Parameters |
| :--- | :--- | :--- | :--- |
| **Input Layer** | $128 \times 128 \times 3$ Tensors | - | Direct Medical Preprocessed Arrays |
| **Conv Block 1** | 32 Filters ($3 \times 3$ Kernel) | ReLU | `BatchNormalization()` + `MaxPooling2D` ($2 \times 2$) + Spatial `Dropout` (0.25) |
| **Conv Block 2** | 64 Filters ($3 \times 3$ Kernel) | ReLU | `BatchNormalization()` + `MaxPooling2D` ($2 \times 2$) + Spatial `Dropout` (0.25) |
| **Conv Block 3** | 128 Filters ($3 \times 3$ Kernel) | ReLU | `BatchNormalization()` + `MaxPooling2D` ($2 \times 2$) + Spatial `Dropout` (0.25) |
| **Conv Block 4** | 256 Filters ($3 \times 3$ Kernel) | ReLU | `BatchNormalization()` + `MaxPooling2D` ($2 \times 2$) + Spatial `Dropout` (0.25) |
| **Flattening** | - | - | Transforms Multi-Dimensional Spatial Vectors to 1D Array |
| **Dense Head 1** | 512 Units | ReLU | Deep Dense Fully-Connected Layer + Structured `Dropout` (0.50) |
| **Dense Head 2** | 256 Units | ReLU | Intermediate Latent Representation Head + Structured `Dropout` (0.50) |
| **Output Head** | 3 Units | Softmax | Outputs Categorical Probability Vectors for (Normal, Tumor, Stroke) |

#### Structural Highlights:
* **Batch Normalization Cascade:** Placed immediately post-convolution to continuously normalize activations, mitigating internal covariate shift and speeding up training convergence.
* **Dual-Layer High Dropout:** The final classification blocks use a steep $50\%$ Dropout rate, forcing the network to develop redundant pathways and preventing heavy overfitting on synthetic artifacts.

---

## ⚙️ Optimization Strategy & Callbacks
* **Objective Function:** `SparseCategoricalCrossentropy` for multi-class optimization without needing one-hot encoding overhead.
* **Optimization Algorithm:** `Adam` Optimizer initialized with a default learning rate of $\alpha = 0.001$.
* **Adaptive Callbacks:**
  1. `EarlyStopping`: Monitored on validation loss with a patience threshold of 5 epochs. If validation loss stops improving, training terminates immediately to save the best model weights.
  2. `ReduceLROnPlateau`: Monitored on validation loss. If the loss plateaus for 3 successive epochs, the learning rate drops by an adjustment factor of $0.2$ ($\alpha \rightarrow \alpha \times 0.2$), allowing the model to smoothly converge into deep local minima.

---

## 📊 Analytical Diagnostics & Performance Evaluation

The model's evaluation module bypasses simple accuracy scores, deploying a full diagnostic testing framework via `scikit-plot` and `seaborn`:
1. **Dynamic Training Histories:** Outputs loss and accuracy trends over time to ensure validation metrics closely mirror training progress without overfitting gaps.
2. **Clinical Confusion Matrix:** Quantifies exact pixel-level prediction errors, calculating target misclassifications between structural tumor pixels and ischemic stroke bands.
3. **Multi-Class ROC Curves:** Evaluates True Positive Rates (TPR) against False Positive Rates (FPR) across changing prediction thresholds, logging both Micro and Macro-averaged Area Under the Curve (AUC).
4. **Precision-Recall Profiles:** Evaluates model precision under structured constraints, outputting individual **Precision, Recall, and F1-Scores** for every single clinical class.

---

## 🛠️ Computational Requirements & Dependencies
Before initializing the notebook script, ensure your computing environment has the following dependencies configured:
```bash
pip install tensorflow opencv-python matplotlib seaborn scikit-plot scikit-learn
