# Deep Learning for Computer Vision

A comprehensive collection of deep learning projects and implementations focused on computer vision tasks. This repository contains practical implementations ranging from basic neural networks built from scratch to advanced CNN architectures for various computer vision applications.

## 📚 Table of Contents

- [Overview](#overview)
- [Technology Stack](#technology-stack)
- [Repository Structure](#repository-structure)
- [Projects](#projects)
  - [Fundamentals](#fundamentals)
  - [Image Classification](#image-classification)
  - [Medical Image Analysis](#medical-image-analysis)
  - [Regression](#regression)
- [Getting Started](#getting-started)
- [Prerequisites](#prerequisites)
- [Learning Resources](#learning-resources)

## 🎯 Overview

This repository serves as a comprehensive learning journey through deep learning concepts and applications in computer vision. It includes implementations of neural networks from scratch, tensor operations, and various CNN architectures for real-world computer vision tasks.

The projects primarily follow the tutorial by freeCodeCamp: [**"Deep Learning for Computer Vision with Python and TensorFlow"**](https://www.youtube.com/watch?v=IA3WxTTPXqQ)

## 🛠 Technology Stack

- **Deep Learning Frameworks:** TensorFlow, PyTorch
- **Scientific Computing:** NumPy, Pandas
- **Data Visualization:** Matplotlib, Seaborn
- **Computer Vision:** OpenCV (where applicable)
- **Development Environment:** Jupyter Notebooks, Python

## 📁 Repository Structure

```
deep-learning/
├── scratch/                    # Neural network from scratch implementation
├── mnist/                      # MNIST digit recognition projects
├── cifar/                      # CIFAR-10 image classification
├── eye-cataract/              # Medical image classification
├── data/                      # Datasets
├── *.ipynb                    # Individual project notebooks
└── README.md                  # This file
```

## 🚀 Projects

### Fundamentals

#### 🧠 Neural Network from Scratch
- **File:** `scratch/network.py`
- **Description:** Complete implementation of a neural network using only NumPy, including:
  - Forward propagation
  - Backpropagation algorithm
  - Stochastic Gradient Descent (SGD)
  - Sigmoid activation function
- **Learning Goals:** Understanding the mathematical foundations of neural networks

#### 🔢 TensorFlow Tensors Deep Dive
- **File:** `deep_learning_tensors.ipynb`
- **Description:** Comprehensive exploration of TensorFlow tensor operations including:
  - Tensor creation and manipulation
  - Mathematical operations
  - Shape transformations
  - Broadcasting
- **Learning Goals:** Mastering TensorFlow's fundamental data structures

### Image Classification

#### 📊 MNIST Digit Recognition
- **Files:** 
  - `mnist-recognition.ipynb` - Basic approach
  - `mnist/cnn.ipynb` - CNN implementation
- **Description:** Classic handwritten digit classification using:
  - Basic neural networks
  - Convolutional Neural Networks (CNNs)
  - Performance comparison between approaches
- **Learning Goals:** Understanding CNN architecture and its advantages

#### 🖼️ CIFAR-10 Classification
- **Files:**
  - `cifar/cifar-10-cnn.ipynb` - Custom CNN architecture
  - `cifar/cifar-10-finetuning-vgg-16.ipynb` - Transfer learning with VGG-16
- **Description:** Multi-class image classification with:
  - Custom CNN design
  - Transfer learning techniques
  - Model fine-tuning strategies
- **Learning Goals:** Advanced CNN architectures and transfer learning

#### 🐱🐶 Cat vs Dog Classification
- **File:** `cat-dog-classification.ipynb`
- **Description:** Binary image classification project
- **Learning Goals:** Binary classification techniques and data preprocessing

### Medical Image Analysis

#### 👁️ Eye Cataract Detection
- **File:** `eye-cataract/eye-cataract-classification.ipynb`
- **Description:** Medical image classification for cataract detection
- **Learning Goals:** Applying deep learning to medical imaging

#### 🦠 Malaria Cell Classification
- **File:** `malaria_classification.ipynb`
- **Description:** Classification of malaria-infected blood cells
- **Learning Goals:** Medical image analysis and class imbalance handling

### Regression

#### 🚗 Car Price Prediction
- **File:** `car_price_prediction.ipynb`
- **Dataset:** `data/second-hand-car.csv`
- **Description:** Regression analysis for predicting used car prices
- **Learning Goals:** Applying neural networks to regression problems

## 🏁 Getting Started

### Prerequisites

- Python 3.7+
- Jupyter Notebook or JupyterLab
- GPU support recommended for training (optional)

### Installation

1. Clone the repository:
```bash
git clone https://github.com/abdulhakkeempa/deep-learning.git
cd deep-learning
```

2. Install required packages:
```bash
pip install tensorflow torch torchvision numpy pandas matplotlib seaborn jupyter
```

3. Launch Jupyter Notebook:
```bash
jupyter notebook
```

4. Open any `.ipynb` file to start exploring!

### Running the Projects

Each notebook is self-contained and can be run independently. For the neural network from scratch:

```bash
cd scratch
python network.py
```

## 📋 Prerequisites

### Knowledge Requirements
- Basic Python programming
- Understanding of linear algebra (vectors, matrices)
- Basic calculus (derivatives)
- Familiarity with machine learning concepts

### Recommended Background
- Introduction to Machine Learning
- Basic statistics and probability
- Familiarity with NumPy and Pandas

## 📖 Learning Resources

### Essential Learning Materials
1. [CNN Explainer](https://poloclub.github.io/cnn-explainer/) - Interactive CNN visualization
2. [ConvNet JS](https://cs.stanford.edu/people/karpathy/convnetjs/demo/mnist.html) - Neural network playground
3. [Pooling Layers](https://machinelearningmastery.com/pooling-layers-for-convolutional-neural-networks/) - Understanding pooling operations

### Additional Resources
- [Deep Learning Specialization by Andrew Ng](https://www.coursera.org/specializations/deep-learning)
- [CS231n: Convolutional Neural Networks for Visual Recognition](http://cs231n.stanford.edu/)
- [TensorFlow Official Tutorials](https://www.tensorflow.org/tutorials)
- [PyTorch Tutorials](https://pytorch.org/tutorials/)

### Books
- *Deep Learning* by Ian Goodfellow, Yoshua Bengio, and Aaron Courville
- *Hands-On Machine Learning* by Aurélien Géron
- *Python Machine Learning* by Sebastian Raschka

---

**Happy Learning! 🚀**

Feel free to explore the notebooks, experiment with the code, and adapt the implementations for your own projects!
