#!/usr/bin/env python
# coding: utf-8

# In[5]:


def predictT(row, weights):
  activation = weights[0]
  for i in range(len(row)-1):
    activation += weights[i + 1] * row[i]
  return 1.0 if activation >= 0.0 else 0.0
 
def train_weightsT(train, l_rate, n_epoch):
  weights = [0.0 for i in range(len(train[0]))]
  for epoch in range(n_epoch):
    sum_error = 0.0
    for row in train:
      prediction = predictT(row, weights)
      error = row[-1] - prediction
      sum_error += error**2
      weights[0] = weights[0] + l_rate * error
      for i in range(len(row)-1):
        weights[i + 1] = weights[i + 1] + l_rate * error * row[i]
    print('>epoch=%d, lrate=%.3f, error=%.3f' % (epoch, l_rate, sum_error))
  return weights
 
dataset_AND = [[0, 0, 0],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1]]

dataset_OR = [[0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 1]]

dataset_XOR = [[0, 0, 0],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]]

dataset_NOT = [[0, 1],
            [1, 0]]

dataset_NAND = [[0, 0, 1],
            [1, 0, 1],
            [0, 1, 1],
            [1, 1, 0]]

dataset_NOR = [[0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 0]]

dataset_XNOR = [[0, 0, 1],
            [1, 0, 0],
            [0, 1, 0],
            [1, 1, 1]]

l_rate = 0.1
n_epoch = 5
weights1 = train_weightsT(dataset_AND, l_rate, n_epoch)
weights2 = train_weightsT(dataset_OR, l_rate, n_epoch)
weights3 = train_weightsT(dataset_XOR, l_rate, n_epoch)
weights4 = train_weightsT(dataset_NOT, l_rate, n_epoch)
weights5 = train_weightsT(dataset_NAND, l_rate, n_epoch)
weights6 = train_weightsT(dataset_NOR, l_rate, n_epoch)
weights7 = train_weightsT(dataset_XNOR, l_rate, n_epoch)

print(weights1)
print(weights2)
print(weights3)
print(weights4)
print(weights5)
print(weights6)
print(weights7)

print("AND Gate")
for row in dataset_AND:
  prediction = predictT(row, weights1)
  print("Expected=%d, Predicted=%d" % (row[-1], prediction))

print("OR Gate")
for row in dataset_OR:
  prediction = predictT(row, weights2)
  print("Expected=%d, Predicted=%d" % (row[-1], prediction))

print("XOR Gate")
for row in dataset_XOR:
  prediction = predictT(row, weights3)
  print("Expected=%d, Predicted=%d" % (row[-1], prediction))


# In[ ]:




