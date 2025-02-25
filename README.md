# MNIST Digit Classification

## Overview
The **MNIST Digit Classification** project is designed to classify handwritten digits (0-9) using a **3-layer neural network**. The model is built using **TensorFlow** and other essential Python libraries for data analysis and visualization. The neural network achieves an impressive **99.7% accuracy**.

## Features
- **Data Analysis & Visualization:** Utilizes NumPy, Pandas, Matplotlib, and Seaborn for data exploration.
- **Deep Learning Model:** A 3-layer neural network with optimized architecture.
- **High Accuracy:** Achieves 99.7% accuracy in classifying digits.
- **Softmax Probability Prediction:** Provides probability distribution over 10 classes.

## Technologies Used
- **NumPy & Pandas** (for data handling and preprocessing)
- **Matplotlib & Seaborn** (for data visualization)
- **TensorFlow** (for building and training the neural network)

## Neural Network Architecture
- **Input Layer:** MNIST image (28x28 pixels, flattened to 784 input nodes)
- **Hidden Layer 1:** 25 neurons, **ReLU activation**
- **Hidden Layer 2:** 15 neurons, **ReLU activation**
- **Output Layer:** 10 neurons (one for each digit), **Linear activation**
- **Softmax Activation:** Converts output to probability distribution

## Installation & Setup
### Prerequisites
Ensure you have Python installed. You can download it from [Python's official website](https://www.python.org/downloads/).

### Step 1: Clone the Repository
```bash
git clone https://github.com/SayandipSaha666/Handwritten-Digit-Classification.git
cd Handwritten-Digit-Classification
```

### Step 2: Install Required Libraries
Install the dependencies using pip:
```bash
pip install numpy pandas matplotlib seaborn tensorflow
```

### Step 3: Run the Model
Execute the script to train and evaluate the neural network:
```bash
python mnist_classification.py
```

## Working Mechanism
1. **Load Data:** MNIST dataset is loaded and preprocessed.
2. **Build Neural Network:** A 3-layer neural network is constructed using TensorFlow.
3. **Train Model:** The model is trained using an optimized loss function and optimizer.
4. **Predict & Evaluate:** The model predicts digit classes and achieves a **99.7% accuracy**.

## Project Structure
```
Handwritten-Digit-Classification/
│── dataset/                 # Contains MNIST dataset
│── models/                  # Saved trained models
│── mnist_classification.py  # Main script for training & evaluation
│── requirements.txt         # Required dependencies
│── README.md                # Project Documentation
```

## Future Enhancements
- Implement **Convolutional Neural Networks (CNNs)** for improved accuracy.
- Optimize hyperparameters using **Grid Search or Bayesian Optimization**.
- Deploy the model as a **web application** for real-time digit classification.

## Contributors
- **Sayandip Saha** (GitHub: [SayandipSaha666](https://github.com/SayandipSaha666))

## License
This project is licensed under the **MIT License**. Feel free to modify and use it.

---
For any issues or suggestions, please open an issue in the [GitHub Repository](https://github.com/SayandipSaha666/Handwritten-Digit-Classification).

