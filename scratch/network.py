import random
import numpy as np

class NeuralNetwork:
  def __init__(self, sizes):
    self.sizes = sizes
    self.num_layers = len(sizes)
    self.biases = [np.zeros((y, 1)) for y in sizes[1:]]
    self.weights = [np.random.randn(y, x) * np.sqrt(1 / x) for x, y in zip(sizes[:-1], sizes[1:])]
    self.threshold = 0.5

  def feedforward(self, x):
    for b, w in zip(self.biases, self.weights):
      x = self.sigmoid(np.dot(w, x) + b)
    return x
  
  def sigmoid(self, z):
    return 1.0/(1.0 + np.exp(-z))

  def SGD(self, training_data, epochs, lr, batch_size):
    no_of_training_data = len(training_data)

    for epoch in range(epochs):
      random.shuffle(training_data)

      batches = [
        training_data[k: k + batch_size]
        for k in range(0, no_of_training_data, batch_size)
      ]

      for batch in batches:
        self.batch_update(batch, lr)

      print(f"{epoch} completed.")


  def batch_update(self, batch, lr):
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]

    for x,y in batch:
      delta_nabla_w, delta_nabla_b = self.backpropagate(x, y)
      nabla_w = [nb_w + dn_w for nb_w, dn_w in zip(nabla_w, delta_nabla_w)]
      nabla_b = [nb_b + dn_b for nb_b, dn_b in zip(nabla_b, delta_nabla_b)]


    self.weights = [w - ((lr/len(batch)))*n_w for w, n_w in zip(self.weights, nabla_w)]
    self.biases = [b - ((lr/len(batch)))*n_b for b, n_b in zip(self.biases, nabla_b)]

  def backpropagate(self, x, y):
    nabla_w = [np.zeros(w.shape) for w in self.weights]
    nabla_b = [np.zeros(b.shape) for b in self.biases]

    activation = x
    activations = []
    activations.append(x)

    zs = []

    for w, b in zip(self.weights, self.biases):
      z = np.dot(w, activation) + b
      zs.append(z)

      activation = self.sigmoid(z)
      activations.append(activation)

    delta = self.cost_derivative(activations[-1], y) * self.sigmoid_prime(zs[-1])

    nabla_b[-1] = delta
    nabla_w[-1] = np.dot(delta, activations[-2].transpose())

    for l in range(2, self.num_layers):
      z = zs[-l]
      sp = self.sigmoid_prime(z)

      delta = np.dot(self.weights[-l + 1].transpose(), delta) * sp

      nabla_b[-l] = delta
      nabla_w[-l] = np.dot(delta, activations[-l - 1].transpose())

    return nabla_w, nabla_b

  def cost_derivative(self, activation, y):
    return (activation - y)

  def sigmoid_prime(self, z):
    sig = self.sigmoid(z)
    return sig*(1 - sig)

nn = NeuralNetwork([2,3,1])

training_data = [
    (np.array([[0.0], [0.0]]), np.array([[0.0]])),
    (np.array([[0.0], [1.0]]), np.array([[1.0]])),
    (np.array([[1.0], [0.0]]), np.array([[1.0]])),
    (np.array([[1.0], [1.0]]), np.array([[0.0]]))
]


print(nn.weights)
print(nn.biases)

nn.SGD(
  training_data= training_data,
  epochs=30,
  lr=0.05,
  batch_size=2
)

print(nn.weights)
print(nn.biases)

for train_data in training_data:
  print(train_data[0])
  print(nn.feedforward(train_data[0]))

