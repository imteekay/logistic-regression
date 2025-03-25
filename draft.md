# Logistic Regression

## Classification

- Questions
  - Is this email spam? no | yes
  - Is the transaction fraudulant? no | yes
  - Is the tumor malignant? no | yes
- `y` can only be one of the two values
  - binary classification
- Terminology
  - class = category = label
  - false = 0 (negative class)
  - true = 1 (positive class)

## Classification algorithm

- Based on features, classify if the value is one label or the other
- e.g. based on the tumor size, classify if the cancer is malignant or benign
- For classification, a linear function doesn't work, we need a threshold to say if the predicted value is one label or the other
- The decision boundary of the logistic regression is when the threshold meets the model function. The left side of the boundary will be classified as one label and the right side, the other one.

## Logistic Regression

The logistic function fits the data and build the curve and the algorithm outputs the threshold to separate the labels

![](./images/001.png)

A classification algorithm can use a logistic function, also called the sigmoid function. A function with an S-shape that outputs between 0 and 1

![](./images/002.png)

The logistic regression algorithm computes the linear combination z = xw + b and pass it to the sigmoid activation function

![](./images/003.png)

The interpretation of the logistic regression output

- It outputs the probability that class/label is 1
- e.g.
  - x = tumor size
  - y = 0 for benign
  - y = 1 for malignant
  - f(x) = 0.7 -> 70% chance that y = 1 (cancer is malignant)
- P(y = 0) + P(y = 1) = 1 (100%)
