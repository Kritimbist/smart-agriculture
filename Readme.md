# 🌾 Krishi Sathi - Plant Disease Detection Algorithm

[![Python](https://img.shields.io/badge/Python-3.8+-blue.svg)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.0+-red.svg)](https://pytorch.org/)
[![License](https://img.shields.io/badge/License-MIT-green.svg)](LICENSE)
[![Accuracy](https://img.shields.io/badge/Accuracy-95.6%25-brightgreen.svg)]()

> **AI-Powered Plant Disease Detection System using Deep Residual Learning with Class-Balanced Training**

A comprehensive deep learning solution for automated plant disease classification, specifically designed to handle severe class imbalance (36:1 ratio) in agricultural datasets. Built as part of the Krishi Sathi: Smart Farmer Nepal platform at Tribhuvan University.

---

## 📋 Table of Contents

- [Overview](#overview)
- [Algorithm Architecture](#algorithm-architecture)
- [Key Features](#key-features)
- [Mathematical Foundation](#mathematical-foundation)

---

## 🎯 Overview

### Problem Statement

Agricultural datasets suffer from severe class imbalance where common diseases have 36× more training samples than rare diseases, causing models to be biased toward frequent classes. Traditional approaches fail to detect rare but critical diseases.

### Our Solution

We developed **ResNet-9**, a lightweight deep residual convolutional neural network with:
- **Dual-balancing strategy**: Weighted loss + weighted sampling
- **Residual connections**: Prevents vanishing gradients
- **Test-Time Augmentation**: 40% variance reduction
- **95.6% accuracy** across all 38 disease classes

### Key Innovation

```python
# Class Weight Calculation (addresses 36:1 imbalance)
w_i = N / (C × n_i)

where:
  N = Total samples (54,305)
  C = Number of classes (38)
  n_i = Samples in class i
  
Result: Rare diseases get 9.4× higher importance than common ones
```

---

## 🏗️ Algorithm Architecture

### ResNet-9 Network Structure

```
INPUT: 224×224×3 RGB Image
   ↓
┌─────────────────────────────────────────────┐
│ Conv Block 1: 3→64 channels                 │
│   • 3×3 Convolution                         │
│   • Batch Normalization                     │
│   • ReLU Activation                         │
└─────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────┐
│ Conv Block 2: 64→128 channels + MaxPool     │
│   • 3×3 Convolution                         │
│   • Batch Normalization                     │
│   • ReLU Activation                         │
│   • 2×2 Max Pooling                         │
│   • Dropout (0.1)                           │
└─────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────┐
│ Residual Block 1: 128→128 channels          │
│   ┌───────────────────┐                     │
│   │ Conv 3×3          │                     │
│   │ BatchNorm + ReLU  │                     │
│   │ Conv 3×3          │                     │
│   │ BatchNorm + ReLU  │                     │
│   └───────────────────┘                     │
│          │                                   │
│          └──────────(+)← Identity Skip      │
└─────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────┐
│ Conv Block 3: 128→256 channels + MaxPool    │
└─────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────┐
│ Conv Block 4: 256→512 channels + MaxPool    │
└─────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────┐
│ Residual Block 2: 512→512 channels          │
│   (same structure as ResBlock 1)            │
└─────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────┐
│ Global Average Pooling: 512×28×28 → 512     │
└─────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────┐
│ Fully Connected 1: 512→256                  │
│   • Dropout (0.4)                           │
│   • Batch Normalization                     │
│   • ReLU Activation                         │
│   • Dropout (0.3)                           │
└─────────────────────────────────────────────┘
   ↓
┌─────────────────────────────────────────────┐
│ Output Layer: 256→38 (disease classes)      │
│   • Softmax Activation                      │
└─────────────────────────────────────────────┘
   ↓
OUTPUT: Disease Prediction + Confidence Score
```

**Total Parameters:** 11.2 Million  
**Model Size:** 43 MB  
**Inference Time:** 30ms (single) | 150ms (TTA)

---

## ✨ Key Features

### 1. **Class Imbalance Handling**

```python
# Weighted Cross-Entropy Loss
L = -w_y × Σ ỹ_i × log(p_i)

# Weighted Random Sampling
sample_weight = 1 / class_sample_count
sampler = WeightedRandomSampler(weights, num_samples)
```

**Impact:** Reduced prediction bias from 3.9× to <2×

### 2. **Residual Learning**

```python
# Skip Connection Formula
output = F(x) + x

where F(x) = learned transformation
      x = identity mapping
```

**Impact:** Enables training of 9-layer network without vanishing gradients

### 3. **Test-Time Augmentation**

```python
# TTA Ensemble
predictions = []
for transform in [original, hflip, rotate_10, rotate_-10, center_crop]:
    pred = model(transform(image))
    predictions.append(pred)

final_prediction = mean(predictions)
```

**Impact:** 40% reduction in prediction variance

### 4. **Smart Training Strategy**

- **Optimizer:** AdamW (weight_decay=1e-4)
- **LR Schedule:** OneCycle (max_lr=0.001)
- **Regularization:** Dropout, BatchNorm, Label Smoothing
- **Early Stopping:** Patience=10 epochs

---

## 📐 Mathematical Foundation

### Forward Propagation

```python
# Convolutional Layer
Z = W * X + b                    # Feature extraction
Ẑ = (Z - μ) / √(σ² + ε)         # Batch normalization
A = max(0, Ẑ)                    # ReLU activation

# Output Layer
z = W_out · A + b_out            # Logits
p_i = exp(z_i) / Σ exp(z_j)      # Softmax probabilities
ŷ = argmax(p_i)                  # Final prediction
```

### Backward Propagation

```python
# Loss Gradient (Output Layer)
∂L/∂z_i = p_i - y_i

# Chain Rule (Hidden Layers)
∂L/∂W_l = ∂L/∂a_(l+1) × ∂a_(l+1)/∂z_l × ∂z_l/∂W_l

# Weight Update (AdamW)
m_t = β₁·m_(t-1) + (1-β₁)·g_t                    # Momentum
v_t = β₂·v_(t-1) + (1-β₂)·g_t²                   # Variance
W_t = W_(t-1) - η·(m_t/√(v_t+ε) + λ·W_(t-1))   # Update
```

### Class Weight Formula

```python
w_i = N / (C × n_i)

Example:
  Potato_healthy: w = 54,305/(38×152) = 9.402
  Orange_disease: w = 54,305/(38×5,507) = 0.260
```
