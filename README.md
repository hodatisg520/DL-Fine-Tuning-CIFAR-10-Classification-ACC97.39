# Deep Learning Fine-Tuning for CIFAR-10 Classification

This repository demonstrates advanced deep learning techniques for image classification, focusing on model efficiency and high-performance fine-tuning. The project achieves state-of-the-art accuracy on the CIFAR-10 dataset while strictly maintaining a lightweight architecture of under 4 million parameters.

## Project Overview

The objective of this project is to build, fine-tune, and evaluate deep neural networks under parameter constraints. Rather than relying on massive, computationally expensive models, this implementation leverages a heavily optimized, ultra-lightweight version of EfficientNet-B0. By employing advanced training strategies, data augmentation, and regularization, the model achieves exceptional generalization on unseen data.

## Dataset

- **Dataset:** CIFAR-10
- **Composition:** 60,000 32x32 color images across 10 classes (50,000 training images, 10,000 test images).
- **Preprocessing:** Images are resized to 128x128 and normalized using standard ImageNet statistics to ensure training stability and accelerate convergence.

## Architecture and Methodology

To comply with the strict constraint of utilizing fewer than 4 million parameters while maximizing accuracy, the following architectural and training choices were implemented:

1. **Model Selection:** `EfficientNet-B0` was chosen as the base model due to its optimal balance of accuracy and parameter efficiency.
2. **Architecture Modification (Ultra-Lightweight):** The base architecture was modified into `EfficientNetB0_CIFAR_UltraLight`. By removing the final expansion layers and adapting the classification head (using an Adaptive Average Pooling layer followed by a Linear classifier), the model was successfully reduced to **3,598,598 parameters**, well below the 4M limit.
3. **Loss Function:** `CrossEntropyLoss` with Label Smoothing (0.1) was utilized. This reduces the model's confidence in its predictions, preventing overfitting and improving generalization.
4. **Optimizer and Scheduler:** `AdamW` optimizer with a learning rate of 4e-4 and a weight decay of 2e-5 was used alongside a `CosineAnnealingLR` scheduler to smoothly navigate the loss landscape.

## Advanced Data Augmentation

To prevent the model from memorizing the training dataset and to ensure robust feature extraction, rigorous data augmentation strategies were employed:
- **AutoAugment:** Utilizing Google Brain's optimal policy for CIFAR-10.
- **Random Erasing:** Randomly removing sections of the input images to force the model to learn broader context rather than relying on specific localized features.

## Results

Despite the heavy constraints on parameter size, the fine-tuned model achieved outstanding performance:

- **Validation Accuracy:** ~97.19%
- **Test Time Augmentation (TTA):** By applying TTA (averaging predictions of original and horizontally flipped images), the test accuracy further improved to **97.39%**.

The combination of a highly efficient architecture, state-of-the-art data augmentation, and careful hyperparameter tuning proves that compact deep learning models can perform competitively with significantly larger architectures.

## Repository Contents

- `ML4ds_assignment_5_deeplearning_2.ipynb`: The main Jupyter Notebook containing the full training pipeline, from data preparation to model definition, training loop, and final evaluation.
