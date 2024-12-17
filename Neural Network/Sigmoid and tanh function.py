#!/usr/bin/env python
# coding: utf-8

# In[1]:


#Sigmoid activation function
import math

def sigmoid(x):
    return 1.0 / (1.0 + math.exp(-x))


def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if sigmoid(activation) >= 0.5 else 0.0


def train_weights(train, learning_rate, n_epoch):
    weights = [0.0 for _ in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:

            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error ** 2

            weights[0] += learning_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] += learning_rate * error * row[i]

        if epoch % 100 == 0 or epoch == n_epoch - 1:
            print(f'>epoch={epoch+1}, lrate={learning_rate:.3f}, error={sum_error:.3f}')

    return weights

dataset1 = [
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]

learning_rate = 0.1
n_epoch = 5

weights = train_weights(dataset1, learning_rate, n_epoch)

print("Final weights:", weights)

print("\nTest Predictions:")
for row in dataset1:
    prediction = predict(row, weights)
    print(f"Expected={row[-1]}, Predicted={prediction}")


# In[2]:


#Tanh activation function
def exp_approx(x, terms=20):
    result = 1.0
    term = 1.0
    for n in range(1, terms):
        term *= x / n
        result += term
    return result

def tanh(x):
    e_x = exp_approx(x)
    e_neg_x = exp_approx(-x)
    return (e_x - e_neg_x) / (e_x + e_neg_x)

def predict(row, weights):
    activation = weights[0]
    for i in range(len(row) - 1):
        activation += weights[i + 1] * row[i]
    return 1.0 if tanh(activation) >= 0.5 else 0.0

def train_weights(train, learning_rate, n_epoch):
    weights = [0.0 for _ in range(len(train[0]))]
    for epoch in range(n_epoch):
        sum_error = 0.0
        for row in train:
            prediction = predict(row, weights)
            error = row[-1] - prediction
            sum_error += error ** 2
            weights[0] += learning_rate * error
            for i in range(len(row) - 1):
                weights[i + 1] += learning_rate * error * row[i]
        if epoch % 100 == 0 or epoch == n_epoch - 1:
            print(f'>epoch={epoch+1}, lrate={learning_rate:.3f}, error={sum_error:.3f}')

    return weights


dataset1 = [
    [0, 0, 0],
    [1, 0, 1],
    [0, 1, 1],
    [1, 1, 1],
]

learning_rate = 0.1
n_epoch = 2

weights = train_weights(dataset1, learning_rate, n_epoch)

print("Final weights:", weights)

print("\nTest Predictions:")
for row in dataset1:
    prediction = predict(row, weights)
    print(f"Expected={row[-1]}, Predicted={prediction}")


# In[ ]:




