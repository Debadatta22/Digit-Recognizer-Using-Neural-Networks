# Digit-Recognizer-Using-Neural-Networks
Internship based task by Proxenix about Data Science and Analytics for Digit Recognizer Using Neural Networks

#  Handwritten Digit Recognizer Using Neural Networks
**📌 Objective:**

The goal of this project was to build a basic neural network from scratch using only NumPy to recognize handwritten digits from the widely-used MNIST dataset (available on Kaggle). Instead of using machine learning libraries like TensorFlow or PyTorch, I wanted to explore and implement all the core concepts manually to strengthen my understanding of how neural networks work at the lowest level.

**Vision:**

The vision behind this project was to gain deep conceptual clarity and hands-on experience with:

Neural network architecture

Training through forward and backward propagation

Matrix operations for weights and biases

Image classification using raw pixel data

Building smart systems from scratch with basic tools

# Introduction:

Handwritten digit recognition is a classic beginner machine learning project. This task involves taking 28x28 grayscale images of digits (0 through 9) and classifying them correctly. The dataset I used is a CSV file where each row contains:

A label (the actual digit),

Followed by 784 pixel values (28x28 = 784) representing the image.

This project does not use pre-built deep learning libraries. Instead, I coded everything manually using:

Python

NumPy

Matplotlib (for visualization)
# Dataset Used: train.csv (from Kaggle)
**What's in this dataset?**
This file contains handwritten digits (like '0' to '9'), written by different people.

Each image is:

28 x 28 pixels = 784 pixels

Stored as a row of 785 columns:

1st column → Label (which digit it is: 0 to 9)

Next 784 columns → Pixel values (0 to 255, where 0 = black, 255 = white)

**Goal of the Code:**

🎯 Train a neural network to recognize digits (0–9) from the pixel values of the images.

# Project description

As part of my practical learning journey in machine learning and deep learning, I undertook a hands-on project titled "Handwritten Digit Recognition Using Neural Networks". The core objective of this project was to build a digit recognition model from scratch—without relying on advanced machine learning libraries—by implementing the entire logic using Python and NumPy. The dataset used for training and evaluation was sourced from Kaggle’s MNIST digit recognition dataset, which contains thousands of 28x28 pixel grayscale images of handwritten digits ranging from 0 to 9.

The training dataset was preprocessed by normalizing the pixel values and separating the labels (actual digits) from the pixel data. I implemented a basic feedforward neural network with two layers: the first layer had 10 neurons (hidden layer), and the second layer had 10 output neurons representing the 10 possible digit classes. The network used the ReLU (Rectified Linear Unit) function as the activation function in the hidden layer, and the Softmax function in the output layer to convert raw outputs into class probabilities.

One of the key learning experiences in this project was implementing the training process through forward propagation, error calculation using categorical cross-entropy, and backward propagation using derivatives of the activation functions. Weights and biases were updated iteratively using gradient descent to minimize the loss and improve accuracy. Throughout the training process, I monitored the model’s performance and accuracy at regular intervals.

Finally, I tested the trained model by predicting digits from the test set and visualizing the predictions alongside actual labels. The model successfully recognized handwritten digits with good accuracy, and this project helped me develop a deeper understanding of how neural networks function internally. It was a valuable experience in applying mathematical concepts to build real-world AI solutions.

#  Explanation of my code

In this project, I built a digit recognizer using a neural network from scratch using only Python and NumPy, without using any machine learning libraries like TensorFlow or PyTorch. The goal was to understand how neural networks work internally by implementing each step manually. The dataset I used was the popular MNIST handwritten digit dataset (from Kaggle), which contains thousands of grayscale images of digits (0–9), each of size 28x28 pixels. These images are flattened into 784 pixel values (one per column), and the first column contains the correct digit label for each image.

I started the project by loading the dataset using pandas and then converting it into a NumPy array for easier manipulation. I shuffled the data randomly to avoid any order bias and then split it into two parts: one for training and one for development (testing). Each image was normalized by dividing pixel values by 255 so that they fall between 0 and 1, which is crucial for better performance during training.

The neural network I designed has two layers:

The first layer (hidden layer) has 10 neurons and takes the 784 input pixels.

The second layer (output layer) also has 10 neurons, representing the 10 possible digits (0–9).

To start training, I initialized weights and biases randomly for both layers. In the forward propagation step, I used the ReLU (Rectified Linear Unit) activation function in the first layer, which passes only positive values, and the Softmax function in the output layer to convert the final scores into probabilities across the 10 digit classes.

Then, I implemented backpropagation, where the model calculates how much each parameter (weight and bias) contributes to the error and updates them using gradient descent. I also used one-hot encoding to represent the correct output format for comparing predictions.

The model was trained over several iterations. During each iteration, it made predictions, calculated how far the predictions were from the actual labels, and adjusted the weights and biases accordingly to improve accuracy. I also printed out the accuracy after every few steps to monitor how well the model was learning.

Finally, I wrote a function to test the model on individual images. It shows the image using Matplotlib, along with the predicted label and the actual label. When I tested the model on some images, it was able to correctly identify most digits, showing that the model had learned effectively.

This project helped me understand the complete pipeline of neural network training: from data preprocessing, forward and backward propagation, to making predictions and evaluating performance. It gave me practical insights into how AI models learn from data and how we can build intelligent systems from scratch using only basic math and logic.


---

## 🔧 Code Workflow & Explanation

### 1. Load & Prepare Data
- Read CSV using pandas
- Shuffle dataset for randomness
- Transpose data for easier manipulation
- Normalize pixel values (divide by 255)

### 2. Split into Training & Testing Sets
- First 1000 samples for validation (Dev set)
- Remaining for training

### 3. Neural Network Architecture
- **Input Layer**: 784 neurons (28x28 pixels)
- **Hidden Layer**: 10 neurons with ReLU
- **Output Layer**: 10 neurons with Softmax (digit classes 0–9)

### 4. Forward Propagation
- Calculate activations using matrix multiplication
- Apply ReLU and Softmax for hidden/output layers

### 5. Loss Function & One-Hot Encoding
- Use categorical cross-entropy loss
- Convert label digits to one-hot encoded vectors

### 6. Backward Propagation
- Compute gradients of weights & biases
- Apply derivative of ReLU & Softmax
- Chain rule is used to backpropagate error

### 7. Gradient Descent
- Update weights and biases using learning rate (`alpha`)
- Loop over multiple iterations (500) to improve accuracy

### 8. Prediction & Visualization
- `make_predictions()` generates predicted digit
- `test_prediction()` displays the image with actual and predicted label using `matplotlib`

---

## 🧠 Key Concepts Covered
- Neural Networks from Scratch
- Forward and Backward Propagation
- ReLU and Softmax Activation Functions
- Gradient Descent Optimization
- One-Hot Encoding
- NumPy Matrix Manipulation
- Data Normalization
- Image Classification

---

## 💡 Sample Results




# 🔚 Conclusion:
This project gave me practical experience in building a working digit classifier using fundamental principles of machine learning. By manually implementing each part of the neural network, I was able to understand how deep learning models function internally and how they learn to make intelligent predictions from image data.

It was a rewarding exercise that boosted my confidence in AI fundamentals and strengthened my ability to solve real-world problems using mathematical and programming knowledge.

