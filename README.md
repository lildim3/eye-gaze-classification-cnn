# Eye Gaze Classification with CNN

This repository contains a convolutional neural network (CNN) developed for multi-class eye-gaze direction classification from images. The project focuses on classifying eye orientation into one of eight gaze-direction categories using a supervised deep learning pipeline with data augmentation, convolutional feature extraction, and regularization techniques.

## Project Overview

The goal of this project is to design and train a CNN model for image classification on an eye-gaze dataset. The dataset contains **61,073 JPG images**, divided into **8 classes** representing the direction of eye gaze:

- Bottom Left
- Bottom Right
- Bottom Center
- Top Left
- Top Right
- Top Center
- Middle Left
- Middle Right

The dataset is intended for gaze prediction and driver inattention analysis, where robust recognition of eye direction is important for attention monitoring systems.

## Main Objective

The main objective of the project is to classify images of eyes into one of eight gaze-direction classes using a convolutional neural network with strong generalization performance.

## Dataset Characteristics

The dataset contains a large number of labeled eye images with a relatively balanced class distribution. Since the eye orientation itself is the key feature for classification, standard geometric augmentations such as arbitrary rotations were avoided because they could alter the semantic meaning of gaze direction.

Instead, the model uses augmentation strategies that preserve gaze orientation while still improving generalization, such as:

- random zoom
- random contrast adjustment
- random brightness adjustment

These transformations help the model become more robust to visual variability without changing the true class label.

## Model Architecture

The neural network is implemented as a **Sequential CNN model** and consists of the following layers:

- Conv2D
- Batch Normalization
- MaxPooling2D
- Conv2D
- Batch Normalization
- MaxPooling2D
- Conv2D
- Batch Normalization
- MaxPooling2D
- Dropout
- Flatten
- Dense
- Dense

### Total number of parameters
**614,440**

The largest number of parameters comes from the fully connected layers near the output, while the convolutional layers are responsible for hierarchical feature extraction.

## Activation Functions and Optimization

The model uses:

- **ReLU** activation in the hidden layers  
- **Softmax** activation in the output layer  
- **Adam** optimizer for training

### Why these choices?

- **ReLU** helps avoid saturation issues and reduces the risk of vanishing gradients compared to sigmoid or tanh.
- **Softmax** is appropriate for multi-class classification because it outputs class probabilities.
- **Adam** combines momentum and adaptive learning-rate updates, making training efficient and stable.

## Regularization and Overfitting Prevention

To reduce overfitting, the following techniques are used:

- **Batch Normalization**
- **Dropout**
- **Early Stopping**

Early stopping is especially important in this project. Although the maximum number of training epochs was set higher, training stopped earlier when validation performance stopped improving.

## Training Results

The model achieved strong classification performance, with all major evaluation metrics around **0.93**.

### Evaluation metrics

- **Accuracy:** 0.9264
- **Precision:** 0.9270
- **Recall:** 0.9264
- **F1-score:** 0.9263

This means the model correctly classified about **92.64%** of the images and maintained stable performance across all classes.

## Learning Behavior

Training was monitored using both:

- **accuracy curves**
- **loss curves**

Although the maximum number of epochs was set to 30, the training process stopped after the **10th epoch** due to early stopping. After approximately the 5th epoch, validation accuracy no longer improved, which indicated that further training would not significantly improve generalization.

## Confusion Matrix Analysis

The confusion matrices for both training and test sets show strong classification performance. Most values along the main diagonal fall approximately between **0.89 and 0.97**, indicating that the majority of samples were correctly classified in their respective categories.

The similarity between training and test confusion matrices also suggests that the model did not suffer significantly from overfitting.

## Key Features

- multi-class image classification
- CNN-based feature extraction
- 8 gaze-direction classes
- balanced dataset
- orientation-preserving augmentation
- Batch Normalization
- Dropout regularization
- Early Stopping
- Adam optimizer
- strong classification performance

## Suggested Repository Structure

```text
.
├── data/
│   └── dataset_description.txt
├── notebooks/
│   └── training_analysis.ipynb
├── src/
│   ├── model.py
│   ├── train.py
│   ├── evaluate.py
│   └── augmentations.py
├── results/
│   ├── accuracy_loss_curves.png
│   ├── confusion_matrix_train.png
│   └── confusion_matrix_test.png
├── reports/
│   └── CNN_Eye_Gaze_Classification_Report.pdf
└── README.md
