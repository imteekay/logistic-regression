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

The classification prediction should output 0 or 1. The sigmoid or logistic function can be used to map all input values to values between 0 and 1

The logistic function fits the data and build the curve and the algorithm outputs the threshold to separate the labels

![](./images/001.png)

A classification algorithm can use a logistic function, also called the sigmoid function. A function with an S-shape that outputs between 0 and 1

![](./images/002.png)

The logistic regression algorithm computes the linear combination z = xw + b and pass it to the sigmoid activation function.

In numpy, we can use the `exp` function to compute the exponential of a value or an array:

```python
np.exp(1) # 2.718281828459045
np.exp(np.array([1, 2, 3])) # [2.72, 7.39, 20.09]
```

The sigmoid function computes the exponential of the negative value of `z`:

```python
def sigmoid(z):
    """
    Compute the sigmoid of z

    Args:
        z (ndarray): A scalar, numpy array of any size.

    Returns:
        g (ndarray): sigmoid(z), with the same shape as z
         
    """

    return 1 / (1 + np.exp(-z))
```

![](./images/003.png)

The interpretation of the logistic regression output

- It outputs the probability that class/label is 1
- e.g.
  - x = tumor size
  - y = 0 for benign
  - y = 1 for malignant
  - f(x) = 0.7 -> 70% chance that y = 1 (cancer is malignant)
- P(y = 0) + P(y = 1) = 1 (100%)

## Decision Boundary

![](./images/004.png)

- For a sigmoid activation function, the threshold is 0.5
  - For Z >= 0, Y = 1
  - For Z < 0, Y = 0

![](./images/005.png)

- A classification problem where it has two features x1 and x2
  - decision boundary will be a line crossing the graph where it separates the y = 0 and y = 1
- Non-linear decision boundary
  - It can be a polynomial function and not a straight line: circunference, ellipse, cluster/shape

## Cost Function

The cost function in relation to the `w` and `b` forms a non-convex function. But if we try to use gradient descent, there are many "local minima".

![](./images/006.png)

The squared error cost function doesn't work for logistic regression. It isn't as smooth as it's for linear regression. The non-linear nature of the model results in  non-convex cost function with many potential local minima.

Logistic regression requires a cost function more suitable to its non-linear nature.
