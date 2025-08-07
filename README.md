# facial_keypoint_detection

Facial Keypoints Detection
A deep learning solution for detecting facial keypoints using Convolutional Neural Networks (CNN) built with PyTorch and PyTorch Lightning.

Overview

A CNN-based facial keypoints detector built with PyTorch Lightning. Helps protect biometric privacy of users on social networking websites by detecting facial keypoints.

Dataset
The dataset consists of:

Training data: 7,049 face images with corresponding keypoint locations
Test data: 1,783 face images for prediction
Image format: 96x96 grayscale images
Keypoints: 15 facial landmarks with (x,y) coordinates

Model Architecture
The solution uses a Convolutional Neural Network with the following architecture:
Input: 96x96x1 grayscale images
├── Conv2D(64) + BatchNorm + ReLU + MaxPool2D

├── Conv2D(128) + Dropout(0.3) + BatchNorm + ReLU + MaxPool2D

├── Conv2D(256) + Dropout(0.3) + BatchNorm + ReLU + MaxPool2D

├── Conv2D(512) + Dropout(0.3) + BatchNorm + ReLU + MaxPool2D

├── Conv2D(1024) + Dropout(0.3) + BatchNorm + ReLU + MaxPool2D

├── Flatten

├── Linear(1024 → 512) + Dropout(0.3) + BatchNorm1D + ReLU

└── Linear(512 → 30)  # 30 output coordinates

Key Features:

Regularization: Dropout layers (0.3) and Batch Normalization to prevent overfitting
Deep Architecture: 5 convolutional layers for feature extraction
Fully Connected Layers: Dense layers for final coordinate regression
Loss Function: Mean Squared Error (MSE)
Optimizer: Adam with learning rate 1e-3




Training data can be found here - [Training Data](https://drive.google.com/file/d/1c6F1fTUli18nnEItLom7JrwGNDSGfL7c/view?usp=sharing)
Testing data can be found here - [Testing_Data](https://drive.google.com/file/d/1QK4Qy0UblcG3lvyyyIO25P5gpnO2ne4J/view?usp=share_link)