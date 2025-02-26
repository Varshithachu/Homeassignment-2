# Homeassignment-2

Name:Thoutu_Varshith

#1 Cloud Computing for Deep Learning 

Elasticity:It to the ability of a cloud system to automatically adjust resources (such as computing power, storage, and memory) in response to changing workloads. In deep learning, elasticity ensures that resources scale up during model training or inference and scale down when demand decreases, optimizing cost efficiency.

Scalability:It is the capability of a cloud system to handle increasing workloads by adding more resources, either vertically (enhancing the power of existing machines) or horizontally (adding more machines). For deep learning, scalability allows the deployment of large-scale models and parallel processing across distributed GPUs or TPUs.

b)Comparison of AWS SageMaker, Google Vertex AI, and Microsoft Azure Machine Learning Studio

AWS SageMaker

Model Training: Managed Jupyter notebooks, AutoML, and distributed training support.

Infrastructure: Scalable EC2 instances, GPU/TPU support, and serverless inference.

Prebuilt Models: Pretrained models available via AWS Marketplace.

Integration: Works with AWS ecosystem (S3, Lambda, etc.).

Google Vertex AI:

Model Training : Offers AutoML, custom training with TPUs/GPUs, and deep learning containers.

Infrastructure: Uses Google Cloud infrastructure with TPU/GPU support.

Prebuilt Models: Google AI models, including TensorFlow Hub models.

Integration: Integrates with Google Cloud services like BigQuery and Dataflow.

Microsoft Azure ML Studio

Model Training: Supports AutoML, distributed training, and custom environments with MLflow integration.

Infrastructure:Azure VM-based infrastructure with GPU acceleration.

Prebuilt Models:Azure Cognitive Services for pretrained AI models.

Integration:Seamless with Azure services like Power BI and Data Factory.

#2.Convolution Operations with Different Parameters
This repository contains a Python script that demonstrates how to perform convolution operations on a 5x5 input matrix using TensorFlow. The script explores different convolution parameters, such as stride and padding, and visualizes the resulting output feature maps.

## Introduction
Convolution is a fundamental operation in deep learning, particularly in convolutional neural networks (CNNs). This script demonstrates how to perform convolution operations on a 5x5 input matrix using TensorFlow. The script explores different combinations of stride and padding parameters to generate output feature maps.
## Requirements

To run this code, you need the following Python libraries installed:
**NumPy**: A library for numerical computing in Python.
**TensorFlow**: A powerful library for machine learning and deep learning.
You can install this using 
pip install numpy tensorflow
Code Overview:
The script performs the following steps:
Define the Input Matrix: A 5x5 input matrix is defined, which can be modified as needed.
Define the Kernel: A 3x3 kernel is defined for the convolution operation.
Reshape Input and Kernel: The input matrix and kernel are reshaped to match TensorFlow's expected shapes.
Rerform Convolution: A function is defined to perform convolution with specified stride and padding parameters.
Generate Output Feature Maps: The script generates output feature maps for different combinations of stride and padding.

Print Results: The resulting feature maps are printed for each combination of parameters.
Key Functions
tf.nn.conv2d(): Performs the 2D convolution operation using TensorFlow.
perform_convolution(): A helper function to perform convolution with specified stride and padding.
Usage
Clone the repository or download the script.
Ensure you have the required libraries installed.
Run the script using Python:
python convolution_operations.py
The script will print the output feature maps for each combination of stride and padding.
License
This project is licensed under the MIT License

CNN Feature Extraction with Filters and Pooling 
## Introduction
This project demonstrates feature extraction in Convolutional Neural Networks (CNNs) using filters and pooling operations. It applies convolutional filters to extract important features from images and uses pooling layers to reduce dimensionality while preserving essential information.
Importing Required Libraries
The notebook begins by importing the necessary Python libraries, including:
NumPy: For numerical operations.
TensorFlow: For building and applying CNN operations.
Matplotlib: For visualizing images and results.
OpenCV (cv2): For image processing tasks.
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
import cv2
Code Overview:
The notebook consists of the following key sections:
Importing Libraries: Load necessary dependencies for image processing and visualization.
Loading and Preprocessing Images: Read input images and apply necessary preprocessing steps.
Applying Convolutional Filters: Implement edge detection and other feature extraction techniques using convolutional filters.
Pooling Operations: Perform max pooling or average pooling to downsample feature maps.
Visualizing Results: Display intermediate feature maps to understand how CNNs extract information.
Usage
Clone this repository:
git clone(https://github.com/Varshithachu/Homeassignment-2)
Install required dependencies:
pip install numpy matplotlib opencv-python tensorflow
Run the Jupyter Notebook:
jupyter notebook "CNN Feature Extraction with Filters and Pooling.ipynb"
Follow the instructions in the notebook to experiment with different filters and pooling methods.
License
This project is licensed under the MIT License


Implementing and Comparing CNN Architectures 
##Introduction
This project implements the AlexNet architecture, a deep convolutional neural network designed for image classification tasks. The notebook includes steps for building, training, and evaluating the model.
##Requirements:
To run the code in this repository, you need the following:
Python 3.x
TensorFlow 2.x
Keras
You can install the required packages using pip:
pip install tensorflow
##Code Overview:
The notebook includes the following sections:
Importing Libraries: Load necessary dependencies.
Data Preprocessing: Load and preprocess image datasets.
Model Architecture: Define and compile the AlexNet model.
Training the Model: Train the network on a dataset.
Evaluation and Visualization: Evaluate performance and visualize results.
#Usage:
Clone this repository:
git clone (https://github.com/Varshithachu/Homeassignment-2)
Install required dependencies:
pip install tensorflow numpy matplotlib opencv-python
Run the Jupyter Notebook:
jupyter notebook "AlexNet.ipynb"
License:This project is licensed under the MIT License
