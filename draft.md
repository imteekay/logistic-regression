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
- Classification algorithm
  - Based on features, classify if the value is one label or the other
  - e.g. based on the tumor size, classify if the cancer is malignant or benign
  - For classification, a linear function doesn't work, we need a threshold to say if the predicted value is one label or the other
  - The decision boundary of the logistic regression is when the threshold meets the model function. The left side of the boundary will be classified as one label and the right side, the other one.
