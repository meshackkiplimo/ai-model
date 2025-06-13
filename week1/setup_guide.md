# Week 1: Development Environment Setup and ML Fundamentals

## Development Environment Setup

### 1. Python Installation
First, we need to install Python 3.8+ which is recommended for modern ML development.
```bash
# Check if Python is installed
python --version
```

### 2. Virtual Environment
We'll use a virtual environment to manage our project dependencies:
```bash
# Create a new virtual environment
python -m venv ml_env

# Activate the virtual environment
# On Windows:
ml_env\Scripts\activate
# On macOS/Linux:
source ml_env/bin/activate
```

### 3. Essential Libraries
Install these fundamental ML libraries:
```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
pip install numpy pandas matplotlib scikit-learn jupyter
```

## Understanding the Libraries

1. **PyTorch**: Our main deep learning framework
   - Provides tensor operations
   - Neural network building blocks
   - GPU acceleration support

2. **NumPy**: Fundamental package for scientific computing
   - Array operations
   - Mathematical functions
   - Linear algebra operations

3. **Pandas**: Data manipulation and analysis
   - Loading datasets
   - Data preprocessing
   - Statistical operations

4. **Matplotlib**: Data visualization
   - Plotting training curves
   - Visualizing data distributions
   - Model performance graphs

## Basic ML Concepts

### 1. What is Machine Learning?
Machine learning is a subset of artificial intelligence that enables systems to learn and improve from experience without being explicitly programmed.

#### Types of Machine Learning:
- **Supervised Learning**: Learning from labeled data
  - Classification (predicting categories)
  - Regression (predicting continuous values)

- **Unsupervised Learning**: Finding patterns in unlabeled data
  - Clustering
  - Dimensionality reduction

### 2. The Learning Process

```mermaid
graph LR
    A[Data Collection] --> B[Data Preprocessing]
    B --> C[Model Training]
    C --> D[Model Evaluation]
    D --> E[Model Deployment]
```

### 3. Neural Networks Basics

#### Structure of a Neural Network:
```mermaid
graph LR
    I[Input Layer] --> H1[Hidden Layer 1]
    H1 --> H2[Hidden Layer 2]
    H2 --> O[Output Layer]
```

#### Key Components:
1. **Neurons**: Basic computational units
2. **Weights**: Learnable parameters
3. **Activation Functions**: Add non-linearity
   - ReLU
   - Sigmoid
   - Tanh

## First Practice Project

We'll create a simple neural network to understand these concepts:

```python
import torch
import torch.nn as nn

# Define a simple neural network
class SimpleNN(nn.Module):
    def __init__(self):
        super(SimpleNN, self).__init__()
        self.layer1 = nn.Linear(2, 4)  # 2 inputs, 4 hidden units
        self.activation = nn.ReLU()
        self.layer2 = nn.Linear(4, 1)  # 4 hidden units, 1 output
    
    def forward(self, x):
        x = self.layer1(x)
        x = self.activation(x)
        x = self.layer2(x)
        return x

# This network will be explained in detail in our next session
```

## Next Steps

1. Install the required software following this guide
2. Verify your installation by running a simple PyTorch tensor operation
3. Read through the ML concepts
4. Get ready for our first practical implementation

## Questions to Think About

1. Why do we use virtual environments?
2. What's the difference between CPU and GPU computing in ML?
3. How does supervised learning differ from unsupervised learning?
4. Why do neural networks need activation functions?

We'll discuss these questions and start our first implementation in the next session.